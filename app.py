from fastapi.middleware.cors import CORSMiddleware
import io
import pandas as pd
from docx import Document
# pip install python-docx
from PyPDF2 import PdfReader
from fastapi import FastAPI, UploadFile, File, Response
from typing import List
from model import predict_class

origins = ["*"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.options("/")
async def preflight_handler():
    headers = {
        "Access-Control-Allow-Origin": "http://localhost:3000",
        "Access-Control-Allow-Methods": "POST",
        "Access-Control-Allow-Headers": "Content-Type",
    }
    return Response(status_code=200, headers=headers)

@app.post("/class")
async def upload_file(files: List[UploadFile] = File(...)):
    result = []
    for file in files:
        file_extension = file.filename.split('.')[-1]
        file_content = await file.read()

        if file_extension == "pdf":
            pdf_reader = PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

        elif file_extension == "docx":
            document = Document(io.BytesIO(file_content))
            text = " ".join([paragraph.text for paragraph in document.paragraphs])

        elif file_extension in ["xlsx"]:
            try:
                excel_file = pd.read_excel(io.BytesIO(file_content))
                text = excel_file.to_string(index=False)
            except Exception as e:
                print(e)
                text = "can't predict class"
        else:
            text = "can't predict class"
        result.append({"fileName": file.filename, "class": predict_class(text)})
    print(result)
    return result
