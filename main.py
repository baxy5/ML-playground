import asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File

from agent.config import process_pdf_file

app = FastAPI()


@app.post("/summary")
async def summarize_and_key_concepts(file: UploadFile = File()):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only PDF files are allowed."
        )

    try:
        # answer = await
        answer = ""
    except Exception as e:
        return {"error": str(e)}

    return {"message": answer}
