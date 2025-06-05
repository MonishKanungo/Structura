from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import asyncio
import uuid
import pandas as pd
from typing import List, Optional
import pickle

# Import the main processing and classifier logic from final1.py
from final1 import process_pdf_table, DataframeIntentClassifier

app = FastAPI()

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_images"
EXTRACTED_DIR = "extracted_tables"
CLASSIFIER_PATH = "classifier.pkl"

# In-memory state
state = {
    "classifier": None,
    "output_dir": EXTRACTED_DIR,
    "image_folder": UPLOAD_DIR,
    "last_upload_id": None,
    "processing_status": {
        "status": "idle",  # idle, processing, completed, error
        "progress": 0,
        "message": "",
        "error": None
    }
}

class QARequest(BaseModel):
    question: str

class QAResponse(BaseModel):
    answer: str
    selected_table: Optional[str] = None
    table_title: Optional[str] = None
    table_description: Optional[str] = None
    sample_data: Optional[list] = None

def update_processing_status(status, progress=0, message="", error=None):
    """Update the global processing status"""
    state["processing_status"] = {
        "status": status,
        "progress": progress,
        "message": message,
        "error": error
    }
    return state["processing_status"]

@app.post("/upload_images/")
async def upload_images(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Upload images for processing.
    This endpoint will start background processing of the uploaded images.
    """
    try:
        # Reset status
        update_processing_status("uploading", 0, "Starting upload...")
        
        # Create upload directory if it doesn't exist
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        os.makedirs(EXTRACTED_DIR, exist_ok=True)
        
        # Clear previous uploads
        update_processing_status("uploading", 5, "Cleaning up previous uploads...")
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        
        # Save new uploads
        update_processing_status("uploading", 10, "Saving uploaded files...")
        saved_files = []
        for i, file in enumerate(files):
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)
            progress = 10 + int((i + 1) / len(files) * 30)  # Up to 40% for upload
            update_processing_status("uploading", progress, f"Uploaded {i+1}/{len(files)} files...")
        
        # Generate a unique ID for this upload
        upload_id = str(uuid.uuid4())
        state["last_upload_id"] = upload_id
        
        # Start background processing
        update_processing_status("processing", 40, "Starting document processing...")
        background_tasks.add_task(process_and_store_classifier, UPLOAD_DIR)
        
        return {"message": "Files uploaded and processing started", "upload_id": upload_id}
    except Exception as e:
        update_processing_status("error", 0, "Error during upload", str(e))
        raise HTTPException(status_code=500, detail=str(e))

async def process_and_store_classifier(image_folder):
    """Process the uploaded images and store the classifier"""
    try:
        update_processing_status("processing", 45, "Analyzing document structure...")
        
        # Get list of image files
        image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
        
        if not image_files:
            update_processing_status("error", 0, "No valid image files found")
            return False
        
        update_processing_status("processing", 50, f"Processing {len(image_files)} pages...")
        
        # Process the PDF/Images and extract tables
        classifier = await process_pdf_table(
            pdf_path=None,  # We're using images
            original_input_image_paths=image_files,
            output_dir=EXTRACTED_DIR
        )
        
        if classifier:
            update_processing_status("processing", 90, "Saving results...")
            # Save the classifier
            with open(CLASSIFIER_PATH, 'wb') as f:
                pickle.dump(classifier, f)
            state["classifier"] = classifier
            update_processing_status("completed", 100, "Processing complete!")
            return True
            
        update_processing_status("error", 0, "Failed to process files")
        return False
        
    except Exception as e:
        error_msg = f"Error processing files: {str(e)}"
        print(error_msg)
        update_processing_status("error", 0, "Processing failed", error_msg)
        return False

@app.post("/qa/", response_model=QAResponse)
async def qa(request: QARequest):
    """
    Ask a question about the most recently uploaded and processed tables.
    """
    classifier = state.get("classifier")
    if classifier is None:
        raise HTTPException(status_code=400, detail="No tables available. Please upload images first.")
    # Use the classifier to select the most relevant table
    result = classifier.answer_query(request.question)
    if not result or "selected_df_name" not in result:
        return QAResponse(answer="No relevant table found for your question.")
    selected_table = result["selected_df_name"]
    df = result["dataframe"]
    # Compose prompt for Gemini
    import google.generativeai as genai
    llm = genai.GenerativeModel("gemini-2.0-flash")
    csv_data = df.to_csv(index=False)
    gemini_prompt = f"""
You are a data expert. You are given a table in CSV format below. Answer the user's question using only the data in the table. If the answer is not present, say so.

User question: {request.question}

Table data (CSV):
{csv_data}
"""
    response = llm.generate_content(gemini_prompt)
    return QAResponse(
        answer=response.text,
        selected_table=selected_table,
        table_title=result["metadata"].get("title", "Untitled Table"),
        table_description=result["metadata"].get("description", "No description available"),
        sample_data=df.head(5).to_dict(orient="records")
    )

@app.get("/status")
async def status():
    """
    Get the status of the current processing job.
    """
    status_info = {
        "status": state["processing_status"]["status"],
        "progress": state["processing_status"]["progress"],
        "message": state["processing_status"]["message"],
        "classifier_loaded": state["classifier"] is not None,
        "last_upload_id": state["last_upload_id"]
    }
    
    if state["processing_status"]["error"]:
        status_info["error"] = state["processing_status"]["error"]
        
    return status_info
