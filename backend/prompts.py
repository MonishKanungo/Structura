# prompts.py

# Prompt for extracting headers from table images
EXTRACT_HEADERS_PROMPT = '''
You are an expert at extracting tables from images with complex nested headers.
Analyze the given image and return only the table headers in JSON format.

Return a single JSON object with the key "headers" only.

If there are nested headers, combine them using " > " between levels (e.g., "MainHeader > SubHeader").

If headers are not nested, include them directly.

Ensure the headers are in the same left-to-right order as they appear in the image.

Do not return title, description, or any explanations.

Do not extract any table data — only the headers.

The output must be valid, parsable JSON.
Do not include ```json or ``` markers in the response

Return only the JSON — no Markdown, no extra text.
For example, if there's a header "Size" with sub-headers "Width" and "Length", you would return:
"headers": ["Size > Width", "Size > Length"]
'''

# Prompt for extracting table data from text and image
EXTRACT_DATA_PROMPT = '''You are an expert at extracting table data from text. You're given:
1. Text extracted from a PDF containing table data
2. An image of the table (as backup reference only)
3. The already-identified column headers: {headers_str}

PRIORITY INSTRUCTION: Your primary source of data should be the TEXT. Use the image only for reference if the text is ambiguous.

Task: Extract ONLY the table data rows from the TEXT, matching it to the provided headers. Return a JSON object with:
- "data": 2D array where each inner array (row) has exactly the same number of elements as there are headers

Rules for data extraction:
- Use the provided headers
- DO NOT modify or reinterpret the headers
- CRITICAL: You MUST (this is compulsory) do the following for every cell in every row:
    1. Preserve all dash ('-') values exactly as they appear - do NOT replace them with null or empty values.
    2. Convert all mixed fractions (for example: "3⅛", "3¼", "3⅜", "3½", "3⅝", "3¾", "3⅞") into decimal equivalents:
       - Example conversions: 3⅛ → 3.125, 3¼ → 3.25, 3⅜ → 3.375
    3. This conversion is MANDATORY for every cell where a mixed fraction appears.
- Make sure each row has EXACTLY the same number of columns as there are headers
- Focus on extracting structured data that appears to be in tabular format from the text
- Look for clear patterns of data that align with the number of headers provided
- Convert numeric values properly:
  * "5,960" should be 5960 (remove commas)
  * If a value appears to be a decimal missing a leading zero (like .813), format it as 0.813
  * If a value appears to be a decimal missing the point but clearly meant as such (like 813 in a context needing 0.813), insert the correct decimal point
Special structure rules:
- If you find any cell value like:
  * 3.500" Drill Pipe 13.30 #/Ft.
  * 4.500" Drill Pipe 16.60 #/Ft.
  * 5.500" Casing 17 #/Ft.
  * 7.000" Casing 23 #/Ft. 
  or Similar so then TREAT the entire string as a *single cell value*
-Return ONLY valid, parsable JSON
-Do not include ```json or ``` markers in the response
-DO NOT include any explanations or notes outside the JSON object
The text content from the PDF to extract data from is:{page_text}'''

# Prompt for extracting metadata from table
EXTRACT_METADATA_PROMPT = '''You are an expert at extracting metadata from tables.
Return a JSON object with:
-Title:- Title: {title_str}
- Description: {desc_str}
- "headers": {headers_str}
- "topic_keywords": List of 5-10 keywords that describe the table's subject matter and content
- "data_domain": The general domain this data belongs to (e.g., "oil_drilling", "finance", "medicine", etc.)

Be precise and specific based on the table information provided.do not change key values for title, description, headers
Return ONLY valid, parsable JSON without any explanations.
''' 