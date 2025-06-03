import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, TFAutoModel
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from dotenv import load_dotenv

# Importações locais
from data_loader import load_and_prepare_csv, load_and_prepare_from_pdfs
from embeddings import load_or_compute_embeddings
from model_builder import build_model
import shap_wrapper
from shap_wrapper import compute_shap_values
from report import CreditPDFReport
from grok3_explainer import generate_grok3_response

# Importação do Pinecone
from pinecone_utils import get_or_create_index

# -----------------------------
# Configurações gerais
# -----------------------------
load_dotenv()  # Carrega variáveis do .env (ex: PINECONE_API_KEY)

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Iniciando pipeline de análise de crédito...")

    # 1) Carregar dados com cache
    if os.path.exists(CACHE_DF_PATH):
        logger.info("Carregando DataFrame do cache...")
        df = pd.read_pickle(CACHE_DF_PATH)
    else:
        if os.path.isdir(PDF_FOLDER):
            logger.info(f"Carregando dados a partir de PDFs em '{PDF_FOLDER}'...")
            df = load_and_prepare_from_pdfs(PDF_FOLDER)
        else:
            logger.info(f"Carregando dados a partir do CSV '{CSV_PATH}'...")
            df = load_and_prepare_csv(CSV_PATH)

        df.to_pickle(CACHE_DF_PATH)
        logger.info(f"DataFrame salvo em cache: {CACHE_DF_PATH}")

    # 2) Amostragem e limpeza
    if len(df) > NUM_SAMPLES:
        df = df.sample(n=NUM_SAMPLES, random_state=RANDOM_STATE)
    df = df.reset_index(drop=True)
    sample_indices = df.index.to_numpy()

    df["anos_atividade"] = df["anos_atividade"].fillna(0).astype(int)
    df["rendimento_anual"] = df["rendimento_anual"].fillna(0).clip(lower=0)
    df["divida_total"] = df["divida_total"].fillna(0)
    df["justificativa"] = df["justificativa"].fillna("sem justificativa")

    # 3) Pré‐processamento numérico e label encoding
    num_features = ["anos_atividade", "rendimento_anual", "divida_total"]
    X_num = StandardScaler().fit_transform(df[num_features].astype(np.float32))

    le_porte = LabelEncoder()
    y_porte = le_porte.fit_transform(df["porte_empresa"].astype(str))
    num_portes = len(le_porte.classes_)

    le_risco = LabelEncoder()
    y_risco = le_risco.fit_transform(df["risco_credito"].astype(str))
    y_cat = np.zeros((len(y_risco), len(le_risco.classes_)))
    for i, val in enumerate(y_risco):
        y_cat[i, val] = 1

    logger.info(f"Classes de porte: {list(le_porte.classes_)}")
    logger.info(f"Classes de risco: {list(le_risco.classes_)}")

    # 4) Obtenção de embeddings BERT (com cache interno)
    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    bert_model = TFAutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")
    textos = df["justificativa"].astype(str).tolist()
    X_text = load_or_compute_embeddings(
        textos, tokenizer, bert_model, CACHE_EMBEDDINGS_DIR
    )
    logger.info(f"Dimensão dos embeddings: {X_text.shape}")

    # 5) Indexar embeddings no Pinecone
    try:
        pinecone_index = get_or_create_index(dimension=X_text.shape[1])
        ids = [f"id_{i}" for i in df.index]
        vectors = [(ids[i], X_text[i].tolist()) for i in range(len(ids))]
        pinecone_index.upsert(vectors)
        logger.info("Embeddings enviados ao Pinecone.")
    except Exception as e:
        logger.error(f"Falha ao indexar no Pinecone: {e}")
        # Se quiser continuar mesmo sem Pinecone, comente a linha a seguir
        # return

    # 6) Divisão em treino/teste (uma só vez)
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
        sample_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # 7) Treinamento ou carregamento do modelo
    global shap_wrapper
    from tensorflow.keras.models import load_model

    if os.path.exists(MODEL_PATH):
        logger.info(f"Carregando modelo treinado de '{MODEL_PATH}'...")
        best_model = load_model(MODEL_PATH)
    else:
        logger.info("Inicializando KerasTuner e treinamento do modelo...")
        from keras_tuner import Hyperband

        def tuner_builder(hp):
            return build_model(hp, num_portes, X_num.shape[1], X_text.shape[1])

        tuner = Hyperband(
            tuner_builder,
            objective="val_accuracy",
            max_epochs=5,
            factor=3,
            directory=CACHE_DIR,
            project_name="credito_bert_justif",
        )
        tuner.search(
            [y_porte, X_num, X_text],
            y_cat,
            epochs=5,
            validation_split=TEST_SIZE,
            batch_size=64,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=3),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
            ],
            verbose=1,
        )
        best_model = tuner.get_best_models(num_models=1)[0]

        best_model.fit(
            [Xp_tr, Xn_tr, Xt_tr],
            y_tr,
            epochs=50,
            batch_size=32,
            validation_data=([Xp_te, Xn_te, Xt_te], y_te),
            callbacks=[
                EarlyStopping(
                    monitor="val_loss", patience=4, restore_best_weights=True
                ),
                ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
            ],
            verbose=1,
        )
        best_model.save(MODEL_PATH)
        logger.info(f"Modelo treinado salvo em '{MODEL_PATH}'")

    # Disponibiliza o modelo para o shap_wrapper
    shap_wrapper.best_model = best_model

    # 8) Avaliação no conjunto de teste
    pred_probs = best_model.predict([Xp_te, Xn_te, Xt_te])
    pred_classes = np.argmax(pred_probs, axis=1)
    true_classes = np.argmax(y_te, axis=1)
    logger.info("Relatório de Classificação no Conjunto de Teste:")
    print(
        classification_report(
            true_classes, pred_classes, target_names=le_risco.classes_
        )
    )

    # 9) Explicabilidade SHAP + geração de PDF
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

    pdf.output(OUTPUT_PDF_PATH)
    logger.info(f"Relatório PDF gerado: {OUTPUT_PDF_PATH}")


if __name__ == "__main__":
    main()
