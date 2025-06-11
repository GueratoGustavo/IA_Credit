import torch
import asyncio
import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import AutoTokenizer, TFAutoModel
from tensorflow.keras.models import load_model
import logging
from fpdf import FPDF

# Ajuste para compatibilidade com Torch
torch.classes.__path__ = []
# Policy asyncio para Windows
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configuração de logging
logging.basicConfig(filename="app.log", level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configurações ---
CACHE_DIR = "cache"
CACHE_EMBEDDINGS_DIR = "cache_embeddings"
MODEL_PATH = os.path.join(CACHE_DIR, "best_model.h5")
OUTPUT_PDF_PATH = os.path.join(CACHE_DIR, "relatorio_risco_credito.pdf")

# --- Funções Auxiliares ---


def generate_grok3_response(justificativa, porte, num_features, classe):
    # Formata a explicação em linhas legíveis
    features = ", ".join(f"{k}: {v}" for k, v in num_features.items())
    return {
        "Justificativa": justificativa,
        "Porte da empresa": porte,
        "Características": features,
        "Classe prevista": classe,
        "Análise detalhada da IA": (
            f"Risco classificado como '{classe}' com base nos dados fornecidos."
        )
    }

class CreditPDFReport:
    def __init__(self):
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)

    def add_prediction_page(
        self, idx, classe_predita, prob, justificativa_text, shap_vals, feature_names
    ):
        self.pdf.add_page()
        self.pdf.set_font("Arial", size=12)
        self.pdf.cell(200, 10, txt=f"Análise {idx + 1}", ln=True, align="C")
        self.pdf.cell(200, 10, txt=f"Risco Previsto: {classe_predita}", ln=True)
        self.pdf.cell(200, 10, txt=f"Probabilidade: {max(prob):.2%}", ln=True)
        # converte justificativa dict para texto
        for k, v in justificativa_text.items():
            self.pdf.multi_cell(0, 8, txt=f"{k}: {v}")

    def save_pdf(self, filepath):
        self.pdf.output(filepath)
        logger.info(f"PDF salvo em '{filepath}'")

@st.cache_data
def load_or_compute_embeddings(_texts, _tokenizer, _bert_model, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    inputs = _tokenizer(_texts, return_tensors="tf", padding=True, truncation=True, max_length=128)
    outputs = _bert_model(inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

# --- PIPELINE DE ANÁLISE ---
def pipeline_full_analysis(df):
    st.info("🔍 Iniciando análise dos dados...")
    mapping = {
        "tempo_mercado_anos": "anos_atividade",
        "dividas_totais": "divida_total",
        "porte_empresa": "porte",
        "justificativa": "justificativa"
    }
    df = df.rename(columns=mapping)

    if "rendimento_anual" not in df.columns:
        if "lucro_mensal" in df.columns:
            df["rendimento_anual"] = df["lucro_mensal"] * 12
            st.info("📊 Calculado 'rendimento_anual' a partir de 'lucro_mensal'.")
        elif "receita_mensal" in df.columns:
            df["rendimento_anual"] = df["receita_mensal"] * 12
            st.info("📊 Calculado 'rendimento_anual' a partir de 'receita_mensal'.")
        else:
            st.error(
                "⚠️ 'rendimento_anual' não encontrado. "
                "Forneça 'lucro_mensal' ou 'receita_mensal'."
            )
            st.stop()

    required = ["anos_atividade","rendimento_anual","divida_total","justificativa","porte"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"⚠️ Colunas obrigatórias ausentes: {', '.join(missing)}")
        st.stop()

    def clean_numeric(ser):
        ser = ser.astype(str).str.extract(r"(\d+\.?\d*)")[0]
        return pd.to_numeric(ser, errors="coerce").fillna(0)

    num_feats = ["anos_atividade","rendimento_anual","divida_total"]
    for c in num_feats:
        df[c] = clean_numeric(df[c])
        df[c] = df[c].astype(int if c=="anos_atividade" else float)
        if df[c].min() < 0:
            st.warning(f"⚠️ Valores negativos em '{c}' podem afetar resultados.")

    df[num_feats] = df[num_feats].fillna(0)
    df["justificativa"] = df["justificativa"].fillna("Sem justificativa")
    df["porte"] = df["porte"].fillna("Desconhecido")

    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ Modelo não encontrado em '{MODEL_PATH}'.")
        st.stop()
    model = load_model(MODEL_PATH)
    logger.info("Modelo carregado com sucesso.")

    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    bert = TFAutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")
    texts = df["justificativa"].tolist()
    X_text = load_or_compute_embeddings(texts, tokenizer, bert, CACHE_EMBEDDINGS_DIR)
    X_num = StandardScaler().fit_transform(df[num_feats].astype(np.float32))
    y_port = LabelEncoder().fit_transform(df["porte"])

    probs = model.predict([y_port.reshape(-1,1),X_num,X_text])
    idxs = np.argmax(probs,axis=1)
    labels = np.array(["baixo","médio","alto"])

    results = []
    for i,row in df.iterrows():
        info = generate_grok3_response(
            row["justificativa"],
            row["porte"],
            {nf: row[nf] for nf in num_feats},
            labels[idxs[i]]
        )
        results.append(info)
    df["classe_predita"] = labels[idxs]
    df["explicacao_llm"] = results

    os.makedirs(CACHE_DIR,exist_ok=True)
    pdf = CreditPDFReport()
    for i,info in enumerate(results):
        pdf.add_prediction_page(i,df.loc[i,"classe_predita"],probs[i],info,None,num_feats)
    pdf.save_pdf(OUTPUT_PDF_PATH)
    return df, OUTPUT_PDF_PATH

# --- Streamlit App ---
st.set_page_config(page_title="Análise de Risco de Crédito",layout="wide")

css = ":root{--primary-color:#1A73E8;--secondary-color:#34C759;--text-color:#2C3E50;--bg:#F8FAFC;} body{background:var(--bg);color:var(--text-color);font-family:'Inter',sans-serif;} .stButton>button{background-color:var(--primary-color);color:#fff;border-radius:8px;padding:0.5rem 1.5rem;}"
st.markdown(f"<style>{css}</style>",unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;color:var(--primary-color)'>📊 Análise Inteligente de Risco de Crédito</h1>",unsafe_allow_html=True)
col1,col2 = st.columns([1,2],gap="medium")
with col1:
    st.header("📁 Upload de Dados")
    uploaded = st.file_uploader("Enviar CSV",type=["csv"])
    if st.button("🚀 Rodar Análise"):
        if not uploaded:
            st.error("Envie um CSV")
        else:
            df = pd.read_csv(uploaded)
            df_res,path = pipeline_full_analysis(df)
            st.session_state["df_results"] = df_res
            st.session_state["pdf_path"] = path
            st.success("Análise concluída!")
            st.dataframe(df_res[["classe_predita"]],use_container_width=True)
            with open(path,"rb") as f:
                st.download_button("📥 Baixar PDF",data=f,file_name=os.path.basename(path),mime="application/pdf")
with col2:
    st.header("📊 Resultados")
    if "df_results" in st.session_state:
        st.download_button("📥 PDF Completo",data=open(st.session_state["pdf_path"],"rb"),file_name=os.path.basename(st.session_state["pdf_path"]),mime="application/pdf")
        # Nova estilização da explicação
        info = st.session_state["df_results"].loc[0,"explicacao_llm"]
        exp_html = "<div style='background:var(--primary-color); color:#FFFFFF; padding:1.5rem; border-radius:12px;'>"
        exp_html += "<h3 style='margin-top:0;'>🧠 Explicação da IA</h3>"
        for k,v in info.items():
            exp_html += f"<p style='margin:0.5rem 0;'><strong>{k}:</strong> {v}</p>"
        exp_html += "</div>"
        st.markdown(exp_html,unsafe_allow_html=True)
    else:
        st.info("Envie dados e execute a análise.")
st.markdown("<hr><footer style='text-align:center;color:#6B7280;'>Desenvolvido por IA & Streamlit</footer>",unsafe_allow_html=True)
