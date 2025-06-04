import streamlit as st
import pandas as pd
import os
import shutil
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- IMPORTS SIMULADOS (substitua pelos seus reais) ---
# from your_module import generate_grok3_response, CreditPDFReport, load_or_compute_embeddings, best_model, tokenizer, bert_model
# Vou criar versões mock abaixo só para exemplo:


def generate_grok3_response(justificativa, porte, num_features, classe):
    # Simula a explicação textual gerada pela IA
    return (
        f"Justificativa: {justificativa}\n"
        f"Porte da empresa: {porte}\n"
        f"Características: {num_features}\n"
        f"Classe prevista: {classe}\n"
        f"Análise detalhada da IA: risco classificado como {classe} com base nos dados fornecidos."
    )


class CreditPDFReport:
    def __init__(self):
        self.pages = []

    def add_prediction_page(
        self,
        idx,
        classe_predita,
        prob,
        justificativa_text,
        shap_vals,
        feature_names,
        output_dir,
    ):
        # Simula adição de página no PDF
        self.pages.append(f"Página {idx + 1}: Risco={classe_predita}")

    def save_pdf(self, filepath):
        with open(filepath, "w") as f:
            f.write("\n".join(self.pages))


def load_or_compute_embeddings(texts, tokenizer, bert_model, cache_dir):
    # Mock: retorna array aleatório para embeddings
    return np.random.rand(len(texts), 768)


class BestModelMock:
    def predict(self, inputs):
        # inputs é lista: [porte_encoded, X_num, X_text]
        n = inputs[0].shape[0]
        # Simula probabilidades para 3 classes
        probs = np.random.rand(n, 3)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs


best_model = BestModelMock()
tokenizer = None
bert_model = None
CACHE_EMBEDDINGS_DIR = "./cache_embeddings"
CACHE_DIR = "./cache"

OUTPUT_PDF_PATH = os.path.join(CACHE_DIR, "relatorio_risco_credito.pdf")

# --- Função do pipeline completo ---


def pipeline_full_analysis(df):
    # Preencher e converter colunas
    df["anos_atividade"] = df.get("anos_atividade", 0).fillna(0).astype(int)
    df["rendimento_anual"] = df.get("rendimento_anual", 0).fillna(0).clip(lower=0)
    df["divida_total"] = df.get("divida_total", 0).fillna(0)
    df["justificativa"] = df.get("justificativa", "sem justificativa").fillna(
        "sem justificativa"
    )

    # CORREÇÃO AQUI: Verificar se a coluna 'porte' existe
    if "porte" in df.columns:
        df["porte_empresa"] = df["porte"].fillna("Desconhecido")
    else:
        df["porte_empresa"] = "Desconhecido"

    # Embeddings texto
    textos = df["justificativa"].astype(str).tolist()
    X_text = load_or_compute_embeddings(
        textos, tokenizer, bert_model, CACHE_EMBEDDINGS_DIR
    )

    # Numéricos
    num_features = ["anos_atividade", "rendimento_anual", "divida_total"]
    X_num = StandardScaler().fit_transform(df[num_features].astype(np.float32))

    # Categórico porte
    le_porte = LabelEncoder()
    y_porte = le_porte.fit_transform(df["porte_empresa"].astype(str))

    # Previsão
    pred_probs = best_model.predict([y_porte.reshape(-1, 1), X_num, X_text])
    pred_classes_idx = np.argmax(pred_probs, axis=1)
    classes = np.array(["baixo", "medio", "alto"])  # Defina suas classes reais

    # Explicação LLM
    explicacoes = []
    for i, row in df.iterrows():
        justificativa = row["justificativa"]
        porte_str = row["porte_empresa"]
        num_feats = {
            "anos_atividade": row["anos_atividade"],
            "rendimento_anual": row["rendimento_anual"],
            "divida_total": row["divida_total"],
        }
        classe_predita = classes[pred_classes_idx[i]]

        texto_explica = generate_grok3_response(
            justificativa, porte_str, num_feats, classe_predita
        )
        explicacoes.append(texto_explica)

    df["classe_predita"] = classes[pred_classes_idx]
    df["explicacao_llm"] = explicacoes

    # Gerar PDF (simulado)
    os.makedirs(CACHE_DIR, exist_ok=True)
    pdf = CreditPDFReport()
    for i in range(len(df)):
        pdf.add_prediction_page(
            idx=i,
            classe_predita=df.loc[i, "classe_predita"],
            prob=pred_probs[i],
            justificativa_text=explicacoes[i],
            shap_vals=None,
            feature_names=None,
            output_dir=CACHE_DIR,
        )
    pdf_path = OUTPUT_PDF_PATH
    pdf.save_pdf(pdf_path)

    return df, pdf_path


# --- Streamlit App ---

st.set_page_config(page_title="Análise de Risco de Crédito", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: #2C3E50;'>📊 Análise Inteligente de Risco de Crédito</h1>
    <p style='text-align: center; font-size: 18px; color: #34495E;'>
    Este app utiliza modelos de IA com embeddings BERT, dados estruturados e justificativas textuais
    para prever e explicar o <strong>risco de inadimplência de empresas</strong>.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("📁 Upload de Dados")
    uploaded_csv = st.file_uploader("📄 Enviar arquivo CSV com dados", type=["csv"])

    st.markdown(
        """
        <small style='color:#7F8C8D;'>
        Dica: envie um arquivo CSV contendo os dados financeiros e justificativas para análise.
        </small>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if st.button("🚀 Rodar Análise"):
        if not uploaded_csv:
            st.error("⚠️ Por favor, envie um CSV para começar a análise.")
            st.stop()

        # Limpa cache antigo
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR, exist_ok=True)

        df = pd.read_csv(uploaded_csv)

        df_results, pdf_path = pipeline_full_analysis(df)

        st.success("✅ Análise concluída!")

        st.dataframe(df_results[["classe_predita", "explicacao_llm"]].head())

        with st.expander("🧠 Explicação da IA para a primeira linha", expanded=True):
            st.markdown(
                f"""
                <div style="
                    background-color:#F0F4F8;
                    padding: 15px;
                    border-radius: 8px;
                    border: 1px solid #D1D9E6;
                    font-family: 'Courier New', monospace;
                    font-size: 16px;
                    color:#2C3E50;
                    white-space: pre-wrap;
                ">
                {df_results.loc[0, "explicacao_llm"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📥 Baixar Relatório PDF",
                    data=f,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                    use_container_width=True,
                )

with col2:
    st.header("📊 Resultados")
    if os.path.exists(OUTPUT_PDF_PATH):
        with open(OUTPUT_PDF_PATH, "rb") as f:
            st.download_button(
                label="📥 Baixar Relatório PDF",
                data=f,
                file_name="relatorio_risco_credito.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        st.markdown(
            """
            <p style='color:#27AE60; font-weight:bold;'>
            Relatório gerado com sucesso! Faça o download ou visualize no navegador.
            </p>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "📝 Envie os dados e execute a análise para gerar o relatório com previsões e explicações."
        )

    with st.expander("❓ Sobre o sistema"):
        st.write(
            """
            Este sistema utiliza técnicas avançadas de NLP com embeddings BERT e modelos de deep learning
            para classificar o risco de inadimplência das empresas. O relatório PDF contém as predições,
            explicações interpretáveis (SHAP) e sugestões baseadas em IA.
            """
        )

st.markdown("---")

st.markdown(
    """
    <footer style='text-align:center; color:#95A5A6; font-size:14px;'>
    Desenvolvido com 💡 por IA e Streamlit. Contato: 
    <a href="mailto:guerato.gustavo@gmail.com">guerato.gustavo@gmail.com</a>
    </footer>
    """,
    unsafe_allow_html=True,
)
