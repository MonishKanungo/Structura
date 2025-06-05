import os
import asyncio
import json
import base64
import traceback
import re

import pickle
from PIL import Image
import pandas as pd
import numpy as np

from llama_cloud_services import LlamaParse
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
# from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import create_sql_agent
import getpass
from prompts import EXTRACT_HEADERS_PROMPT, EXTRACT_DATA_PROMPT, EXTRACT_METADATA_PROMPT

# Set API keys
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-9MejFpMZfAtqPWBGhQK5DtE5e3QxyQI4BmdUDNelC1I1m7ET"
GEMINI_API_KEY = "AIzaSyCWpfn1lUYQ8mLevT3vqiSiaM1erUbb4f0"  # Replace with your actual Gemini API key

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# =============== CARRY-FORWARD RULE FOR MERGED CELLS ===============
def apply_carry_forward_rule(df):

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Only process if DataFrame has at least one column
    if len(result_df.columns) == 0:
        return result_df

    first_column = result_df.columns[0]

    # Convert the first column to a list for easier processing
    values = result_df[first_column].tolist()

    # Track the last valid (non-null, non-blank) value for carry-forward
    last_valid_value = None

    for i in range(len(values)):
        current_value = values[i]

        # Check if current value is a dash - PRESERVE IT (no changes)
        if current_value == '-':
            # Dash values stop the carry-forward chain but are preserved as-is
            last_valid_value = '-'  # Dashes can serve as valid values for future carry-forward
            continue

        # Check if current value is truly blank/null/empty
        is_blank = (
            pd.isna(current_value) or
            current_value is None or
            current_value == '' or
            (isinstance(current_value, str) and current_value.strip() == '')
        )

        if is_blank:
            # Only fill blank values if we have a previous valid value
            if last_valid_value is not None:
                values[i] = last_valid_value
            # If no last_valid_value exists, leave as blank
        else:
            # Current value is non-null and non-blank, update carry-forward value
            last_valid_value = current_value

    # Update the DataFrame with processed values
    result_df[first_column] = values

    return result_df

# =============== DATAFRAME INTENT CLASSIFIER ===============
class DataframeIntentClassifier:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the intent classifier with a sentence transformer model.

        Args:
            model_name (str): Name of the sentence-transformers model to use
        """
        self.model = SentenceTransformer(model_name)
        self.dataframes = {}
        self.df_embeddings = {}
        self.dataframe_metadata = {}

    def add_dataframe(self, df, df_name, metadata=None):
        """
        Add a dataframe to the classifier.

        Args:
            df (pd.DataFrame): The dataframe to add
            df_name (str): A unique name/identifier for this dataframe
            metadata (dict, optional): Dictionary containing metadata about the dataframe
                                      including title, description, source, etc.
        """
        # Apply carry-forward rule for merged cells before adding the dataframe
        processed_df = apply_carry_forward_rule(df)
        self.dataframes[df_name] = processed_df

        # Store metadata
        if metadata is None:
            metadata = {}
        self.dataframe_metadata[df_name] = metadata

        # Create text representation of the dataframe headers
        headers = list(processed_df.columns)
        header_text = " ".join(headers)

        # Add title and description if provided
        context_parts = []
        if "title" in metadata and metadata["title"]:
            context_parts.append(metadata["title"])

        if "description" in metadata and metadata["description"]:
            context_parts.append(metadata["description"])

        context_text = " ".join(context_parts)

        # Combine headers and context for embedding
        if context_text:
            embedding_text = f"{header_text} {context_text}"
        else:
            embedding_text = header_text

        # Compute embedding for this dataframe's representation
        embedding = self.model.encode([embedding_text])[0]
        self.df_embeddings[df_name] = embedding

    def classify_query(self, query, top_n=1):
        """
        Classify which dataframe a query is most likely referring to.

        Args:
            query (str): The user query
            top_n (int): Number of top dataframes to return

        Returns:
            list: Top N dataframe names sorted by relevance score
        """
        # Compute embedding for the query
        query_embedding = self.model.encode([query])[0]

        # Calculate similarity scores with all dataframes
        scores = {}
        for df_name, df_embedding in self.df_embeddings.items():
            similarity = cosine_similarity([query_embedding], [df_embedding])[0][0]
            scores[df_name] = similarity

        # Sort dataframes by similarity score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Return top N dataframe names
        return [df_name for df_name, score in sorted_scores[:top_n]]

    def get_dataframe(self, df_name):
        """
        Get a dataframe by name.

        Args:
            df_name (str): Name of the dataframe to retrieve

        Returns:
            pd.DataFrame: The requested dataframe
        """
        return self.dataframes.get(df_name)

    def get_metadata(self, df_name):
        """
        Get metadata for a dataframe by name.

        Args:
            df_name (str): Name of the dataframe to retrieve metadata for

        Returns:
            dict: The metadata for the requested dataframe
        """
        return self.dataframe_metadata.get(df_name, {})

    def answer_query(self, query):
        """
        Get the most relevant dataframe for a query and prepare an answer.

        Args:
            query (str): The user query

        Returns:
            dict: Contains the selected dataframe, its name, and metadata
        """
        if not self.dataframes:
            return {"error": "No dataframes available for querying"}

        top_df_name = self.classify_query(query, top_n=1)[0]
        df = self.get_dataframe(top_df_name)
        metadata = self.get_metadata(top_df_name)

        return {
            "selected_df_name": top_df_name,
            "dataframe": df,
            "metadata": metadata
        }

# =============== UTILITY FUNCTIONS ===============
def encode_image(image_path):
    """Encode an image file to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {str(e)}")
        return None

async def extract_text_from_pdf(pdf_path, image_paths=None):
    """Extract text from PDF or images using LlamaParse"""
    source_type = "input"
    try:
        # Ensure API key is set
        if not os.environ.get("LLAMA_CLOUD_API_KEY"):
            print("Error: LLAMA_CLOUD_API_KEY is not set in environment variables")
            return None
            
        parser = LlamaParse(
            api_key=os.environ["LLAMA_CLOUD_API_KEY"],
            result_type="markdown",
            premium_mode=True
        )
        
        input_to_parse = None
        if pdf_path and os.path.exists(pdf_path):
            input_to_parse = pdf_path
            source_type = "PDF"
        elif image_paths and all(os.path.exists(img) for img in image_paths):
            input_to_parse = image_paths
            source_type = "images"
        else:
            print("Error: Invalid PDF path or image paths provided to extract_text_from_pdf.")
            if pdf_path:
                print(f"PDF path exists: {os.path.exists(pdf_path)}")
            if image_paths:
                print(f"Image paths exist: {[os.path.exists(img) for img in image_paths]}")
            return None

        if not input_to_parse:
            print(f"Error: input_to_parse is None for {source_type} before calling LlamaParse.aparse.")
            return None

        print(f"Attempting to parse {source_type}: {input_to_parse}")
        parsed_output = await parser.aparse(input_to_parse)

        all_pages = []
        if isinstance(parsed_output, list): # Likely for multiple images
            for single_result in parsed_output:
                pages_from_single_result = single_result.get_text_documents(split_by_page=True)
                all_pages.extend(pages_from_single_result)
        elif parsed_output: # Likely for a single PDF or single image result
            all_pages = parsed_output.get_text_documents(split_by_page=True)
        else:
            print(f"LlamaParse returned None or empty for {source_type}: {input_to_parse}")
            return None

        print(f"Successfully extracted {len(all_pages)} pages from {source_type}")
        return all_pages
    except Exception as e:
        print(f"Error extracting text from {source_type}: {str(e)}")
        traceback.print_exc()
        return None

def clean_json_output(json_str):
    """Clean and validate JSON string from LLM output"""
    try:
        # Remove code block markers and clean whitespace
        # Remove any leading/trailing code block markers (``` or ```json or '''json)
        json_str = json_str.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        elif json_str.startswith("'''json"):
            json_str = json_str[7:]
        elif json_str.startswith("```"):
            json_str = json_str[3:]
        elif json_str.startswith("'''"):
            json_str = json_str[3:]
        # Remove any leading 'json' or similar artifact
        if json_str.lstrip().startswith("json"):
            json_str = json_str.lstrip()[4:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        if json_str.endswith("'''"):
            json_str = json_str[:-3]

        json_str = json_str.strip()

        # Additional JSON cleaning steps
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Initial JSON parsing failed: {e}")
            print(f"Problematic JSON: {json_str[:100]}...")

            # Common regex fixes for JSON
            # Fix missing commas between arrays
            json_str = re.sub(r'\]\s*\[', '], [', json_str)

            # Fix trailing commas
            json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)

            # Try one more time
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e2:
                print(f"JSON parsing failed after fixes: {e2}")
                return None

    except Exception as e:
        print(f"Error cleaning JSON: {str(e)}")
        return None

async def extract_headers_with_gemini_vision(image_path):
    """Extract table headers using Gemini Vision API by analyzing the table image"""
    try:
        # Read the image file
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        # Set up Gemini model
        model = genai.GenerativeModel('gemini-2.0-flash')

        prompt = EXTRACT_HEADERS_PROMPT

        # Create content parts for multimodal input
        content_parts = [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(image_data).decode('utf-8')}}
        ]

        # Generate response from Gemini
        response = model.generate_content(content_parts)

        # Extract and clean the output
        raw_response = response.text
        print(f"Raw header extraction response (first 150 chars): {raw_response[:150]}...")

        headers_json = clean_json_output(raw_response)

        if not headers_json:
            print("Failed to extract valid JSON for headers")
            return None

        # Validate the required fields
        if "headers" not in headers_json:
            print("Missing 'headers' field in extracted headers JSON")
            return None

        # Only return the headers list, not extra keys
        headers = headers_json["headers"]
        if not isinstance(headers, list):
            print("Extracted headers is not a list")
            return None

        # Remove any accidental extra keys (like 'headers' as a value)
        headers = [h for h in headers if isinstance(h, str) and h.lower() != "headers"]

        return {"headers": headers}

    except Exception as e:
        print(f"Error extracting headers with Gemini Vision: {str(e)}")
        traceback.print_exc()
        return None

async def extract_data_with_gemini_vision(headers, page_text, image_path):
    """Extract table data using Gemini Vision, using LlamaParse text as reference"""
    try:
        # Read the image file
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        # Set up Gemini model
        model = genai.GenerativeModel('gemini-2.0-flash')

        headers_str = json.dumps(headers)

        prompt = EXTRACT_DATA_PROMPT.format(headers_str=headers_str, page_text=page_text)

        # Create content parts for multimodal input
        content_parts = [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(image_data).decode('utf-8')}}
        ]

        # Generate response from Gemini
        response = model.generate_content(content_parts)

        # Extract and clean the output
        raw_response = response.text
        print(f"Raw data extraction response (first 150 chars): {raw_response[:150]}...")

        data_json = clean_json_output(raw_response)

        if not data_json:
            print("Failed to extract valid JSON for data")
            return None

        # Validate the required fields
        if "data" not in data_json:
            print("Missing 'data' field in extracted data JSON")
            return None

        # Validate data structure - ensure all rows have the same number of columns as headers
        num_headers = len(headers)
        for i, row in enumerate(data_json["data"]):
            if len(row) != num_headers:
                print(f"Row {i} has {len(row)} columns instead of {num_headers}")
                # Fix the row length
                if len(row) < num_headers:
                    row.extend([None] * (num_headers - len(row)))
                else:
                    data_json["data"][i] = row[:num_headers]

        return data_json

    except Exception as e:
        print(f"Error extracting data with Gemini Vision: {str(e)}")
        traceback.print_exc()
        return None

async def extract_metadata_with_gemini(headers, title=None, description=None):
    """Extract additional metadata about the table using Gemini"""
    try:
        # Set up Gemini model
        model = genai.GenerativeModel('gemini-2.0-flash')

        headers_str = ", ".join(headers)
        title_str = title if title else "Unknown Table"
        desc_str = description if description else f"Table with columns: {headers_str}"

        prompt = EXTRACT_METADATA_PROMPT.format(title_str=title_str, desc_str=desc_str, headers_str=headers_str)

        # Generate response from Gemini
        response = model.generate_content(prompt)

        # Extract and clean the output
        raw_response = response.text
        metadata_json = clean_json_output(raw_response)

        if not metadata_json:
            print("Failed to extract valid JSON for metadata")
            return {
                "topic_keywords": [],
                "data_domain": "unknown"
            }

        # Ensure all metadata fields exist
        if "topic_keywords" not in metadata_json:
            metadata_json["topic_keywords"] = []

        if "data_domain" not in metadata_json:
            metadata_json["data_domain"] = "unknown"

        return metadata_json

    except Exception as e:
        print(f"Error extracting metadata with Gemini: {str(e)}")
        return {
            "topic_keywords": [],
            "data_domain": "unknown"
        }

async def convert_pdf_to_images(pdf_path, output_dir="./pdf_images"):
    """Convert each page of PDF to separate images using PyMuPDF"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        image_paths = []
        
        # Convert each page to image
        for i in range(len(doc)):
            # Get the page using the loop variable i
            page = doc[i]  # Fixed: using i instead of undefined page_num
            
            # Convert to image with higher resolution
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
            
            # Save image
            image_path = f"{output_dir}/page_{i+1}.png"
            pix.save(image_path)
            image_paths.append(image_path)
            print(f"Saved page {i+1} image to {image_path}")
        
        return image_paths

    except Exception as e:
        print(f"Error converting PDF to images: {str(e)}")
        traceback.print_exc()
        return []

# =============== MAIN PROCESSING FUNCTIONS ===============

async def process_pdf_table(pdf_path, original_input_image_paths, output_dir="./extracted_tables"):
    """Main function to process PDF and extract tables"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Initialize the intent classifier
        classifier = DataframeIntentClassifier()

        # Step 1: Extract text from PDF or images
        pdf_pages = await extract_text_from_pdf(pdf_path, image_paths=original_input_image_paths)
        if not pdf_pages:
            print("Failed to extract text from input. Aborting process.")
            return None

        # Step 2: Process each page
        for i, page in enumerate(pdf_pages):
            print(f"\nProcessing page {i+1}/{len(pdf_pages)}")

            # Skip if we don't have an image for this page
            if i >= len(original_input_image_paths):
                print(f"No image available for page {i+1}. Skipping.")
                continue

            # Get the image path for this specific page
            page_image_path = original_input_image_paths[i]
            print(f"Using image from: {page_image_path} for page {i+1}")

            # Step 2a: Extract headers using vision LLM
            headers_json = await extract_headers_with_gemini_vision(page_image_path)

            if not headers_json or "headers" not in headers_json:
                print(f"Failed to extract headers from page {i+1}. Skipping.")
                continue

            headers = headers_json["headers"]
            title = headers_json.get("title", f"Table on page {i+1}")
            description = headers_json.get("description", "")

            print(f"Extracted headers: {headers}")
            print(f"Title: {title}")

            # Step 2b: Extract data using vision LLM with reference to LlamaParse text
            data_json = await extract_data_with_gemini_vision(headers, page.text, page_image_path)

            if not data_json or "data" not in data_json:
                print(f"Failed to extract data from page {i+1}. Skipping.")
                continue

            data = data_json["data"]
            print(f"Extracted {len(data)} rows")

            # Step 2c: Extract additional metadata
            metadata_json = await extract_metadata_with_gemini(headers, title, description)
            topic_keywords = metadata_json.get("topic_keywords", [])
            data_domain = metadata_json.get("data_domain", "unknown")

            print(f"Domain: {data_domain}")
            print(f"Keywords: {topic_keywords}")

            # Create DataFrame
            df = pd.DataFrame(data, columns=headers)

            # Apply carry-forward rule for merged cells
            processed_df = apply_carry_forward_rule(df)

            # Print sample of processed data
            print("\nSample processed data (after applying carry-forward rule):")
            print(processed_df.head(3))

            # Prepare metadata for intent classification
            metadata = {
                "title": title,
                "description": description,
                "topic_keywords": topic_keywords,
                "data_domain": data_domain,
                "source_page": i+1,
                "original_headers": headers,  # Store the original column names as extracted
            }

            # Add to classifier
            df_name = f"table_{i+1}_{data_domain}"
            classifier.add_dataframe(processed_df, df_name, metadata)

            processed_filename = f"{output_dir}/{df_name}.csv"
            processed_df.to_csv(processed_filename, index=False)
            print(f"Saved processed data to {processed_filename}")

            # Save metadata separately
            metadata_filename = f"{output_dir}/{df_name}_metadata.json"
            with open(metadata_filename, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved metadata to {metadata_filename}")

        # Save the classifier for later use
        classifier_path = f"{output_dir}/dataframe_classifier.pkl"
        with open(classifier_path, 'wb') as f:
            pickle.dump(classifier, f)
        print(f"\nSaved intent classifier to {classifier_path}")

        print("\nPDF processing completed successfully!")
        return classifier

    except Exception as e:
        print(f"Error in main process: {str(e)}")
        traceback.print_exc()
        return None

def load_classifier(classifier_path):
    """Load a saved classifier from disk"""
    with open(classifier_path, 'rb') as f:
        return pickle.load(f)

def process_all_csvs_in_directory(directory_path, output_directory=None):
    """
    Process all CSV files in a directory, applying the carry-forward rule to each.

    Args:
        directory_path (str): Path to the directory containing CSV files
        output_directory (str, optional): Path to save processed CSV files.
                                         Defaults to same directory with "processed_" prefix.
    """
    if output_directory is None:
        output_directory = directory_path

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv') and not f.startswith('processed_')]

    for csv_file in csv_files:
        input_path = os.path.join(directory_path, csv_file)
        output_path = os.path.join(output_directory, f"{csv_file}")

        try:
            # Load and process the CSV
            df = pd.read_csv(input_path)
            processed_df = apply_carry_forward_rule(df)

            # Save the processed DataFrame
            processed_df.to_csv(output_path, index=False)
            print(f"Processed {csv_file} and saved to {output_path}")
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")

def answer_question(classifier, question):
    """Use the classifier to answer a question about the data"""
    result = classifier.answer_query(question)

    response = {
        "selected_table": result["selected_df_name"],
        "table_title": result["metadata"].get("title", "Untitled Table"),
        "table_description": result["metadata"].get("description", "No description available"),
        "sample_data": result["dataframe"].head(5).to_dict(orient="records")
    }

    return response

def images_to_pdf(folder_path, output_pdf='output.pdf'):
    """
    Convert images in a folder to a PDF file
    
    Args:
        folder_path (str): Path to folder containing images
        output_pdf (str): Name of output PDF file
        
    Returns:
        str: Path to the created PDF file or None if no images found
    """
    image_files = sorted([
        file for file in os.listdir(folder_path)
        if file.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not image_files:
        print("No images found in the folder.")
        return None

    image_list = []
    for file in image_files:
        img_path = os.path.join(folder_path, file)
        img = Image.open(img_path).convert("RGB")
        image_list.append(img)

    output_path = os.path.join(folder_path, output_pdf)
    first_image = image_list[0]
    rest_images = image_list[1:]
    first_image.save(output_path, save_all=True, append_images=rest_images)

    print(f"PDF saved as: {output_path}")
    return output_path

def get_image_paths_from_folder(folder_path):
    """Helper function to get a sorted list of image file paths from a folder."""
    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']
    image_paths = []
    try:
        for f_name in os.listdir(folder_path):
            if os.path.splitext(f_name.lower())[1] in supported_extensions:
                image_paths.append(os.path.join(folder_path, f_name))
        image_paths.sort() # Ensure consistent order
    except FileNotFoundError:
        print(f"Error: Folder not found at {folder_path}")
    except Exception as e:
        print(f"Error listing images in {folder_path}: {e}")
    return image_paths

async def main(image_folder_path, output_dir="./extracted_tables"):
    """
    Main function to process images and extract tables
    
    Args:
        image_folder_path (str): Path to folder containing images
        output_dir (str): Directory to save outputs
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./pdf_images", exist_ok=True)
    
    print(f"Starting PDF table extraction from images in {image_folder_path}")
    print(f"Results will be saved to {output_dir}")
    
    # Step 1a: Get original image paths
    original_image_paths = get_image_paths_from_folder(image_folder_path)
    if not original_image_paths:
        print(f"No images found in {image_folder_path}. Aborting process.")
        return
    print(f"Found {len(original_image_paths)} original images in {image_folder_path}")

    # Step 1b: Convert images to PDF (LlamaParse input)
    # Assuming images_to_pdf takes the folder path and processes images in a sorted order
    # similar to get_image_paths_from_folder to ensure page-image correspondence.
    pdf_path = images_to_pdf(image_folder_path, "combined_images.pdf") 
    if not pdf_path:
        print("Failed to create PDF from images. Aborting process.")
        return
    
    print(f"Created PDF from images: {pdf_path}")
    
    # Step 2: (Removed conversion of PDF back to images)
    # Original images will be used directly for Vision LLM.
    
    # Step 3: Process PDF with LlamaParse and use original images for Vision LLM
    classifier = await process_pdf_table(pdf_path, original_image_paths, output_dir)
    
    if classifier:
        print("\nTable extraction completed successfully!")
        print(f"All extracted tables are saved in {output_dir}")

        # Example query
        test_query = "What data is available in these tables?"
        result = classifier.answer_query(test_query)

        print(f"\nTest Query: '{test_query}'")
        if isinstance(result, dict) and "selected_df_name" in result:
            print(f"Selected DataFrame: '{result['selected_df_name']}'")
            print(f"Table Title: '{result['metadata'].get('title', 'Untitled')}'")
            print("Sample data:")
            print(result['dataframe'].head(3))
        else:
            print("No valid table found or no data extracted.")

        # Save process results summary
        summary = {
            "pdf_path": pdf_path,
            "number_of_pages": len(original_image_paths),
            "number_of_tables": len(classifier.dataframes),
            "table_names": list(classifier.dataframes.keys()),
            "output_directory": output_dir
        }
        
        with open(f"{output_dir}/extraction_summary.json", "w") as f:
            json.dump(summary, f, indent=4)
        
        print(f"\nExtraction summary saved to {output_dir}/extraction_summary.json")

        # === Q&A Integration with LangChain, Gemini, and SQL ===
        try:
            # Set up Gemini API key
            if not os.environ.get("GOOGLE_API_KEY"):
                os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

            # Initialize Gemini LLM
            llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

            # Prompt user for a question
            user_question = input("\nEnter your question about the tables: ")

            # Use the intent classifier to select the most relevant table
            result = classifier.answer_query(user_question)
            if not result or "selected_df_name" not in result:
                print("No relevant table found for your question.")
                return
            selected_table = result["selected_df_name"]
            print(f"\nUsing table for Q&A: {selected_table}")
            df = result["dataframe"]
            print("Columns:", df.columns.tolist())
            print("Shape:", df.shape)

            # Create SQLite DB and load table
            engine = create_engine("sqlite:///table.db")
            df.to_sql("table1", engine, index=False, if_exists="replace")
            db = SQLDatabase(engine=engine)

            # Create SQL agent
            agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=False)

            # Run the agent on the user's question
            answer = agent_executor.invoke({"input": user_question})
            print("\nAnswer:")
            print(answer)
        except Exception as e:
            print(f"Q&A step failed: {e}")
    else:
        print("Table extraction encountered errors. Check the logs for details.")

if __name__== "__main__":
    try:
        # Set API key from the hardcoded value (for development only - move to environment variables in production)
        os.environ["LLAMA_CLOUD_API_KEY"] = "llx-9MejFpMZfAtqPWBGhQK5DtE5e3QxyQI4BmdUDNelC1I1m7ET"
        
        # Path to the folder containing images
        image_folder = "./table_images"  # Change this to your folder path
        
        # Verify the image folder exists
        if not os.path.exists(image_folder):
            print(f"Error: The specified image folder does not exist: {image_folder}")
            exit(1)
            
        print(f"Starting LlamaParse processing on folder: {os.path.abspath(image_folder)}")
        
        # Run the main function
        asyncio.run(main(image_folder))
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
        exit(1)