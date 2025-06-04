from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import os
import shutil
import pandas as pd
from uuid import uuid4

from main import run_credit_analysis  # Função que você vai modularizar
# (ver abaixo)

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/avaliar/")
async def avaliar_credito(file: UploadFile = File(...)):
    filename = f"{uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Salva o arquivo enviado
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Chama a função principal com base no tipo de arquivo
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file.filename.endswith(".pdf"):
            # Se for PDF, passe a pasta (pode ser adaptado se for só 1 PDF)
            os.makedirs("uploads/temp_pdfs", exist_ok=True)
            destination = os.path.join("uploads/temp_pdfs", file.filename)
            shutil.move(file_path, destination)
            df = run_credit_analysis(from_pdf_dir="uploads/temp_pdfs")
        else:
            raise HTTPException(
                status_code=400, 
                detail=(
                    "Formato não suportado (use CSV ou PDF)."
                )
            )

        # Executa pipeline (a função retorna caminho para o PDF gerado)
        output_pdf = run_credit_analysis(dataframe=df)

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erro durante análise: {str(e)}"
        )

    return FileResponse(
        output_pdf, 
        media_type="application/pdf", 
        filename="relatorio_credito.pdf"
    )
