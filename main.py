import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import AutoTokenizer, TFAutoModel
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from data_loader import load_and_prepare_csv, load_and_prepare_from_pdfs
from embeddings import load_or_compute_embeddings
from model_builder import build_model
import shap_wrapper
from shap_wrapper import compute_shap_values
from report import CreditPDFReport
from grok3_explainer import generate_grok3_response
from pinecone_utils import get_or_create_index
from kerastuner import HyperParameters


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_DF_PATH = os.path.join(CACHE_DIR, "cache_df.pkl")
CACHE_EMBEDDINGS_DIR = CACHE_DIR
MODEL_PATH = os.path.join(CACHE_DIR, "best_model.h5")
OUTPUT_PDF_PATH = os.path.join(CACHE_DIR, "relatorio_risco_credito.pdf")
PDF_FOLDER = "pasta_pdfs"
CSV_PATH = "empresas_credito_200k_justificativa.csv"
NUM_SAMPLES = 1000
TEST_SIZE = 0.15
RANDOM_STATE = 42


def train_and_save_model(df):
    logger.info("Iniciando treinamento do modelo...")

    # 1) Preparar features numéricas
    num_features = ["anos_atividade", "rendimento_anual", "divida_total"]
    X_num = StandardScaler().fit_transform(df[num_features].astype(np.float32))

    # 2) LabelEncoder para porte da empresa
    le_porte = LabelEncoder()
    y_porte = le_porte.fit_transform(df["porte_empresa"].astype(str))
    num_portes = len(le_porte.classes_)

    # 3) LabelEncoder para risco de crédito
    le_risco = LabelEncoder()
    y_risco = le_risco.fit_transform(df["risco_credito"].astype(str))
    y_cat = np.eye(len(le_risco.classes_))[y_risco]

    # 4) Embeddings BERT para texto
    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    bert_model = TFAutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")
    textos = df["justificativa"].astype(str).tolist()
    X_text = load_or_compute_embeddings(
        textos, tokenizer, bert_model, CACHE_EMBEDDINGS_DIR
    )
    logger.info(f"Embeddings BERT calculados. Dimensão: {X_text.shape}")

    # 5) Dividir treino/teste
    indices = df.index.values
    (
        Xp_tr,
        Xp_te,
        Xn_tr,
        Xn_te,
        Xt_tr,
        Xt_te,
        y_tr,
        y_te,
        idx_tr,
        idx_te,
    ) = train_test_split(
        y_porte,
        X_num,
        X_text,
        y_cat,
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_risco,
    )

    # 6) Construir modelo com HyperParameters
    hp = HyperParameters()

    model = build_model(
        hp,
        num_portes=num_portes,
        num_features_len=X_num.shape[1],
        text_embedding_dim=X_text.shape[1],
    )

    model.summary(print_fn=logger.info)

    # 7) Callbacks para treino
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)

    # 8) Treinar
    model.fit(
        [Xp_tr, Xn_tr, Xt_tr],
        y_tr,
        validation_data=([Xp_te, Xn_te, Xt_te], y_te),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
    )

    # 9) Salvar modelo treinado
    model.save(MODEL_PATH)
    logger.info(f"Modelo salvo em '{MODEL_PATH}'")


def main():
    logger.info("Iniciando pipeline de análise de crédito...")

    # Carregar dados
    if os.path.exists(CACHE_DF_PATH):
        logger.info("Carregando DataFrame do cache...")
        df = pd.read_pickle(CACHE_DF_PATH)
    else:
        if os.path.isdir(PDF_FOLDER) and len(os.listdir(PDF_FOLDER)) > 0:
            logger.info(f"Carregando dados a partir de PDFs em '{PDF_FOLDER}'...")
            df = load_and_prepare_from_pdfs(PDF_FOLDER)
        else:
            logger.info(f"Carregando dados a partir do CSV '{CSV_PATH}'...")
            df = load_and_prepare_csv(CSV_PATH)
        df.to_pickle(CACHE_DF_PATH)
        logger.info(f"DataFrame salvo em cache: {CACHE_DF_PATH}")

    # Amostragem
    if len(df) > NUM_SAMPLES:
        df = df.sample(n=NUM_SAMPLES, random_state=RANDOM_STATE)
    df = df.reset_index(drop=True)

    # Limpar dados
    df["anos_atividade"] = df["anos_atividade"].fillna(0).astype(int)
    df["rendimento_anual"] = df["rendimento_anual"].fillna(0).clip(lower=0)
    df["divida_total"] = df["divida_total"].fillna(0)
    df["justificativa"] = df["justificativa"].fillna("sem justificativa")

    # Verifica se o modelo existe
    if not os.path.exists(MODEL_PATH):
        logger.info("Modelo treinado não encontrado. Iniciando treino...")
        train_and_save_model(df)

    # Carrega o modelo treinado
    logger.info(f"Carregando modelo treinado de '{MODEL_PATH}'...")
    best_model = load_model(MODEL_PATH)
    shap_wrapper.best_model = best_model

    # Pré-processar e gerar embeddings para predição (mesmo processo do treino)
    num_features = ["anos_atividade", "rendimento_anual", "divida_total"]
    X_num = StandardScaler().fit_transform(df[num_features].astype(np.float32))

    le_porte = LabelEncoder()
    y_porte = le_porte.fit_transform(df["porte_empresa"].astype(str))

    le_risco = LabelEncoder()
    y_risco = le_risco.fit_transform(df["risco_credito"].astype(str))
    y_cat = np.eye(len(le_risco.classes_))[y_risco]

    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    bert_model = TFAutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")

    textos = df["justificativa"].astype(str).tolist()
    X_text = load_or_compute_embeddings(
        textos, tokenizer, bert_model, CACHE_EMBEDDINGS_DIR
    )

    # Divisão treino/teste para avaliação
    indices = df.index.values
    (
        Xp_tr,
        Xp_te,
        Xn_tr,
        Xn_te,
        Xt_tr,
        Xt_te,
        y_tr,
        y_te,
        idx_tr,
        idx_te,
    ) = train_test_split(
        y_porte,
        X_num,
        X_text,
        y_cat,
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_risco,
    )

    # Avaliação no teste
    pred_probs = best_model.predict([Xp_te, Xn_te, Xt_te])
    pred_classes = np.argmax(pred_probs, axis=1)
    true_classes = np.argmax(y_te, axis=1)

    logger.info("Relatório de Classificação no Conjunto de Teste:")
    from sklearn.metrics import classification_report

    print(
        classification_report(
            true_classes, pred_classes, target_names=le_risco.classes_
        )
    )

    # Explicabilidade SHAP + geração PDF (igual seu código)
    logger.info("Iniciando explicabilidade SHAP...")
    shap_values = compute_shap_values(
        Xp_te, Xn_te, Xt_te, background_size=100, nsamples=10
    )

    feature_names = (
        ["porte_empresa"]
        + num_features
        + [f"bert_emb_{i}" for i in range(X_text.shape[1])]
    )

    pdf = CreditPDFReport()
    for i in range(len(Xp_te)):
        probs_i = pred_probs[i]
        pred_idx = np.argmax(probs_i)
        classe_predita = le_risco.classes_[pred_idx]

        original_idx = idx_te[i]
        justificativa_original = df.loc[original_idx, "justificativa"]
        porte_str = df.loc[original_idx, "porte_empresa"]
        num_feats = {
            "anos_atividade": df.loc[original_idx, "anos_atividade"],
            "rendimento_anual": df.loc[original_idx, "rendimento_anual"],
            "divida_total": df.loc[original_idx, "divida_total"],
        }

        explicacao_llm = generate_grok3_response(
            justificativa_original, porte_str, num_feats, classe_predita
        )

        pdf.add_prediction_page(
            idx=i,
            classe_predita=classe_predita,
            prob=probs_i,
            justificativa_text=explicacao_llm,
            shap_vals=shap_values[pred_idx][i],
            feature_names=feature_names,
            output_dir=CACHE_DIR,
        )

    pdf.save_pdf(OUTPUT_PDF_PATH)
    logger.info(f"Relatório PDF gerado em '{OUTPUT_PDF_PATH}'")

    logger.info("Pipeline concluído.")


if __name__ == "__main__":
    main()
