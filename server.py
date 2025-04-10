from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app import report, qa_retrieve_and_answer
import shutil
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

CLIENT_URL = os.getenv("CLIENT_URL")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CLIENT_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Welcome to the API!"}

@app.post("/report")
async def get_data(input_file: UploadFile):
    try:
        if not input_file.filename.endswith(".pdf"):
            return JSONResponse(status_code=400, content={"error": "File must be a PDF."})
        
        os.makedirs("uploads", exist_ok=True)
    
        file_url = f"uploads/{input_file.filename}"
        
        with open(file_url, "wb") as buffer:
            shutil.copyfileobj(input_file.file, buffer)
        
        summary, key_concepts = report(file_url)
        
        if summary and key_concepts:
            return {
                "summary": summary,
                "key_concepts": key_concepts
            }
        else:
            return JSONResponse(status_code=400, content={"error": "Failed to generate report."})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed to generate report. {e}"})
    
@app.post("/chat/{question}")
def chat(question: str):
    answer = qa_retrieve_and_answer(question)
    return {"answer": answer}

#TODO: choose vector database for the project
