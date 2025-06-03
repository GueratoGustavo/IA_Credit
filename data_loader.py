import os
import logging
import re
from typing import Dict

import fitz  # PyMuPDF
import pandas as pd

logger = logging.getLogger(__name__)


def load_and_prepare_csv(csv_path: str) -> pd.DataFrame:
    """
    Carrega um CSV e faz limpeza básica das colunas esperadas.
    """
    df = pd.read_csv(csv_path)
    df["anos_atividade"] = df["anos_atividade"].fillna(0).astype(int)
    df["rendimento_anual"] = df["rendimento_anual"].fillna(0).clip(lower=0)
    df["divida_total"] = df["divida_total"].fillna(0)
    df["justificativa"] = df["justificativa"].fillna("sem justificativa")
    return df


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrai texto de todas as páginas de um arquivo PDF usando PyMuPDF.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Erro ao abrir PDF '{pdf_path}': {e}")
        return ""

    textos = []
    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        textos.append(page.get_text("text"))
    doc.close()
    return "\n".join(textos)


def parse_pdf_record(raw_text: str) -> Dict[str, object]:
    """
    Faz parsing de um texto bruto extraído de PDF para extrair atributos.

    Retorna um dicionário com chaves:
        - anos_atividade (int)
        - rendimento_anual (float)
        - divida_total (float)
        - porte_empresa (str)
        - justificativa (str)
        - risco_credito (str)
    """
    record = {
        "anos_atividade": 0,
        "rendimento_anual": 0.0,
        "divida_total": 0.0,
        "porte_empresa": "desconhecido",
        "justificativa": "sem justificativa",
        "risco_credito": "medio",
    }

    patterns = {
        "anos_atividade": [
            r"Anos de Atividade:\s*(\d+)",
            r"Tempo de atividade\s*\(anos\):\s*(\d+)",
        ],
        "rendimento_anual": [
            r"Rendimento Anual:\s*R\$\s*([\d\.,]+)",
            r"Faturamento Anual:\s*R\$\s*([\d\.,]+)",
        ],
        "divida_total": [
            r"Dívida Total:\s*R\$\s*([\d\.,]+)",
            r"Total de Dívidas:\s*R\$\s*([\d\.,]+)",
        ],
        "porte_empresa": [
            r"Porte da Empresa:\s*([A-Za-zÀ-ÿ0-9 ]+)",
            r"Categoria da Empresa:\s*([A-Za-zÀ-ÿ0-9 ]+)",
        ],
        "justificativa": [
            (
                r"Justificativa:\s*(.*?)\s*"
                r"(?:Anos de Atividade|Risco de Crédito|$)"
            ),
            (
                r"Observação Técnica:\s*(.*?)\s*"
                r"(?:Tempo de atividade|Grau de Risco|$)"
            ),
        ],
        "risco_credito": [
            r"Risco de Crédito:\s*(alto|médio|medio|baixo)",
            r"Grau de Risco:\s*(alto|médio|medio|baixo)",
        ],
    }

    def parse_monetario(m):
        valor_str = m.group(1).replace(".", "").replace(",", ".")
        try:
            return float(valor_str)
        except ValueError:
            return 0.0

    for field, pats in patterns.items():
        for pat in pats:
            match = re.search(
                pat,
                raw_text,
                re.IGNORECASE | (re.DOTALL if field == "justificativa" else 0),
            )
            if match:
                if field == "anos_atividade":
                    record[field] = int(match.group(1))
                elif field in ["rendimento_anual", "divida_total"]:
                    record[field] = parse_monetario(match)
                else:
                    record[field] = match.group(1).strip()
                break

    return record


def load_and_prepare_from_pdfs(pdf_folder: str) -> pd.DataFrame:
    """
    Itera sobre todos os arquivos PDF em 'pdf_folder', extrai e faz parsing,
    retornando um DataFrame com as colunas esperadas.
    """
    registros = []
    for nome in os.listdir(pdf_folder):
        if nome.lower().endswith(".pdf"):
            caminho = os.path.join(pdf_folder, nome)
            texto_raw = extract_text_from_pdf(caminho)
            if texto_raw:
                rec = parse_pdf_record(texto_raw)
                registros.append(rec)
            else:
                logger.warning(f"Texto vazio para PDF: {nome}")

    df = pd.DataFrame(
        registros,
        columns=[
            "anos_atividade",
            "rendimento_anual",
            "divida_total",
            "porte_empresa",
            "justificativa",
            "risco_credito",
        ],
    )
    df["anos_atividade"] = df["anos_atividade"].fillna(0).astype(int)
    df["rendimento_anual"] = df["rendimento_anual"].fillna(0).clip(lower=0)
    df["divida_total"] = df["divida_total"].fillna(0)
    df["justificativa"] = df["justificativa"].fillna("sem justificativa")
    return df
