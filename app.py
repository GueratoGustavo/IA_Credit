import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import fitz
import hashlib
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Dense,
    Flatten,
    Concatenate,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from keras_tuner import Hyperband
from transformers import AutoTokenizer, TFAutoModel
from fpdf import FPDF

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


############################################
## Funções originais (sem alterações)      ##
############################################


def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    df["anos_atividade"] = df["anos_atividade"].fillna(0).astype(int)
    df["rendimento_anual"] = df["rendimento_anual"].fillna(0).clip(lower=0)
    df["divida_total"] = df["divida_total"].fillna(0)
    df["justificativa"] = df["justificativa"].fillna("sem justificativa")
    return df


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text.append(page.get_text("text"))
    doc.close()
    return "\n".join(full_text)


def parse_pdf_to_record(raw_text):
    record = {
        "anos_atividade": 0,
        "rendimento_anual": 0.0,
        "divida_total": 0.0,
        "porte_empresa": "desconhecido",
        "justificativa": "sem justificativa",
        "risco_credito": "medio",
    }

    m_anos1 = re.search(r"Anos de Atividade:\s*(\d+)", raw_text, re.IGNORECASE)
    m_anos2 = re.search(
        r"Tempo de atividade\s*\(anos\):\s*(\d+)", raw_text, re.IGNORECASE
    )
    if m_anos1:
        record["anos_atividade"] = int(m_anos1.group(1))
    elif m_anos2:
        record["anos_atividade"] = int(m_anos2.group(1))

    m_rend1 = re.search(
        r"Rendimento Anual:\s*R\$\s*([\d\.,]+)", raw_text, re.IGNORECASE
    )
    m_rend2 = re.search(
        r"Faturamento Anual:\s*R\$\s*([\d\.,]+)", raw_text, re.IGNORECASE
    )
    if m_rend1:
        rend_str = m_rend1.group(1).replace(".", "").replace(",", ".")
        record["rendimento_anual"] = float(rend_str)
    elif m_rend2:
        rend_str = m_rend2.group(1).replace(".", "").replace(",", ".")
        record["rendimento_anual"] = float(rend_str)

    m_divida1 = re.search(r"Dívida Total:\s*R\$\s*([\d\.,]+)", raw_text, re.IGNORECASE)
    m_divida2 = re.search(
        r"Total de Dívidas:\s*R\$\s*([\d\.,]+)", raw_text, re.IGNORECASE
    )
    if m_divida1:
        div_str = m_divida1.group(1).replace(".", "").replace(",", ".")
        record["divida_total"] = float(div_str)
    elif m_divida2:
        div_str = m_divida2.group(1).replace(".", "").replace(",", ".")
        record["divida_total"] = float(div_str)

    m_porte1 = re.search(
        r"Porte da Empresa:\s*([A-Za-zÀ-ÿ0-9 ]+)", raw_text, re.IGNORECASE
    )
    m_porte2 = re.search(
        r"Categoria da Empresa:\s*([A-Za-zÀ-ÿ0-9 ]+)", raw_text, re.IGNORECASE
    )
    if m_porte1:
        record["porte_empresa"] = m_porte1.group(1).strip()
    elif m_porte2:
        record["porte_empresa"] = m_porte2.group(1).strip()

    m_just1 = re.search(
        r"Justificativa:\s*(.*?)\s*(?:Anos de Atividade|Risco de Crédito|$)",
        raw_text,
        re.DOTALL | re.IGNORECASE,
    )
    m_just2 = re.search(
        r"Observação Técnica:\s*(.*?)\s*(?:Tempo de atividade|Grau de Risco|$)",
        raw_text,
        re.DOTALL | re.IGNORECASE,
    )
    if m_just1:
        record["justificativa"] = m_just1.group(1).strip()
    elif m_just2:
        record["justificativa"] = m_just2.group(1).strip()

    m_risco1 = re.search(
        r"Risco de Crédito:\s*(alto|médio|medio|baixo)", raw_text, re.IGNORECASE
    )
    m_risco2 = re.search(
        r"Grau de Risco:\s*(alto|médio|medio|baixo)", raw_text, re.IGNORECASE
    )
    if m_risco1:
        record["risco_credito"] = m_risco1.group(1).lower().replace("médio", "medio")
    elif m_risco2:
        record["risco_credito"] = m_risco2.group(1).lower().replace("médio", "medio")

    return record


def load_and_prepare_data_from_pdfs(pdf_folder):
    records = []
    for fname in os.listdir(pdf_folder):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(pdf_folder, fname)
            raw_text = extract_text_from_pdf(path)
            rec = parse_pdf_to_record(raw_text)
            records.append(rec)

    df_pdf = pd.DataFrame(
        records,
        columns=[
            "anos_atividade",
            "rendimento_anual",
            "divida_total",
            "porte_empresa",
            "justificativa",
            "risco_credito",
        ],
    )

    df_pdf["anos_atividade"] = df_pdf["anos_atividade"].fillna(0).astype(int)
    df_pdf["rendimento_anual"] = df_pdf["rendimento_anual"].fillna(0).clip(lower=0)
    df_pdf["divida_total"] = df_pdf["divida_total"].fillna(0)
    df_pdf["justificativa"] = df_pdf["justificativa"].fillna("sem justificativa")

    return df_pdf


def get_bert_embeddings(texts, tokenizer, model, max_len=64, batch_size=32):
    embeddings = []
    texts_list = [
        str(t) if not isinstance(t, str) else t
        for t in (texts.tolist() if isinstance(texts, pd.Series) else texts)
    ]

    for i in range(0, len(texts_list), batch_size):
        batch_texts = texts_list[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="tf",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        outputs = model(inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls_embeddings)

    return np.vstack(embeddings)


def build_model(hp, num_portes, num_features_len, text_embedding_dim):
    input_porte = Input(shape=(1,), name="porte_input")
    input_num = Input(shape=(num_features_len,), name="num_input")
    input_text = Input(shape=(text_embedding_dim,), name="text_input")

    emb_dim = hp.Int("embedding_dim", 4, 16, step=4)
    emb_porte = Embedding(input_dim=num_portes + 1, output_dim=emb_dim)(input_porte)
    flat_porte = Flatten()(emb_porte)

    x = Concatenate()([flat_porte, input_num, input_text])

    for i in range(hp.Int("n_layers", 1, 3)):
        units = hp.Int(f"units_{i}", 32, 256, step=32)
        x = Dense(units, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(hp.Float(f"dropout_{i}", 0.2, 0.6, step=0.1))(x)

    output = Dense(3, activation="softmax")(x)
    model = Model(inputs=[input_porte, input_num, input_text], outputs=output)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def model_predict_wrapper(data_numpy):
    porte_part = data_numpy[:, 0].astype(int).reshape(-1, 1)
    num_part = data_numpy[:, 1 : 1 + len(num_features_global)]
    text_part = data_numpy[:, 1 + len(num_features_global) :]
    return best_model.predict([porte_part, num_part, text_part])


class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Relatório de Predição de Risco - Empresa", 0, 1, "C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Página {self.page_no()}", 0, 0, "C")

    def add_prediction(
        self,
        empresa_idx,
        classe_predita,
        prob,
        justificativa_text,
        shap_vals,
        feature_names,
    ):
        self.add_page()
        self.set_font("Arial", size=12)
        self.cell(
            0,
            10,
            f"Empresa {empresa_idx + 1} - Classe Prevista: {classe_predita}",
            ln=True,
        )
        self.cell(0, 10, f"Probabilidades: {np.round(prob, 3)}", ln=True)
        self.cell(0, 10, "Justificativa do Crédito:", ln=True)

        self.multi_cell(0, 10, justificativa_text)

        plt.figure(figsize=(6, 3))
        shap.summary_plot(shap_vals, feature_names=feature_names, show=False)
        plt.tight_layout()
        imagem_path = f"shap_plot_{empresa_idx}.png"
        plt.savefig(imagem_path)
        plt.close()

        self.image(imagem_path, x=10, w=180)
        os.remove(imagem_path)


def generate_grok3_response(justificativa, porte, num_features_dict, classe_predita):
    rendimento_fmt = (
        f"R$ {num_features_dict['rendimento_anual']:,.2f}".replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )
    divida_fmt = (
        f"R$ {num_features_dict['divida_total']:,.2f}".replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )

    texto = (
        f"A empresa foi classificada como '{classe_predita}' com base nos seguintes fatores:\n"
        f"- Porte da empresa ({porte}): empresas de porte {porte} costumam apresentar características que influenciam o risco de crédito.\n"
        f"- Rendimento anual ({rendimento_fmt}): esse valor sugere a capacidade financeira que impacta a classificação.\n"
        f"- Anos de atividade ({num_features_dict['anos_atividade']} anos): indica o nível de experiência e estabilidade.\n"
        f"- Dívida total ({divida_fmt}): reflete o nível de endividamento.\n"
        f'- Justificativa fornecida: "{justificativa.strip()}" foi analisada, apontando fatores adicionais que corroboram a classificação.\n\n'
        f"Com base nesses dados, a classificação '{classe_predita}' é justificada pela combinação de fatores financeiros e operacionais."
    )
    return texto


def gerar_explicacao_llm(justificativa, porte, num_features_dict, classe_predita):
    return generate_grok3_response(
        justificativa,
        porte,
        num_features_dict,
        classe_predita,
    ).strip()


############################################
##   Fim das funções originais            ##
############################################

if __name__ == "__main__":
    print("Iniciando pipeline de análise de crédito com suporte a PDFs e SHAP...\n")

    ### CACHE: diretórios e arquivos de cache ###
    cache_df_path = "cache_df.pkl"  # salva DataFrame completo
    cache_embeddings_path = "cache_X_text.npy"  # salva embeddings BERT
    cache_texts_path = (
        "cache_texts.pkl"  # salva lista de textos originais (justificativas)
    )
    model_path = "best_model.h5"  # salva o modelo treinado
    use_pdf_folder = os.path.isdir(
        "pasta_pdfs"
    )  # verifica se existe pasta 'pasta_pdfs'

    #####################
    # 1) Carregar/Preparo de dados (CSV ou PDFs), com cache
    #####################
    if os.path.exists(cache_df_path):
        print("→ Carregando DataFrame do cache...")
        df = pd.read_pickle(cache_df_path)
    else:
        if use_pdf_folder:
            print("→ Carregando dados a partir de PDFs (pasta_pdfs)...")
            df = load_and_prepare_data_from_pdfs("pasta_pdfs")
        else:
            print("→ Carregando dados a partir de CSV...")
            df = load_and_prepare_data("empresas_credito_200k_justificativa.csv")

        df.to_pickle(cache_df_path)
        print(f"→ DataFrame salvo em cache: {cache_df_path}")

    # Amostrar 1000 amostras aleatórias
    num_samples = 1000
    df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)

    # Garantir que colunas numéricas/texto estejam preenchidas
    df["anos_atividade"] = df["anos_atividade"].fillna(0).astype(int)
    df["rendimento_anual"] = df["rendimento_anual"].fillna(0).clip(lower=0)
    df["divida_total"] = df["divida_total"].fillna(0)
    df["justificativa"] = df["justificativa"].fillna("sem justificativa")

    #####################
    # 2) Pré-processamento numérico e label encoding
    #####################
    num_features_global = ["anos_atividade", "rendimento_anual", "divida_total"]
    X_num = StandardScaler().fit_transform(df[num_features_global].astype(np.float32))

    le_porte = LabelEncoder()
    porte = le_porte.fit_transform(df["porte_empresa"].astype(str))
    num_portes = len(le_porte.classes_)

    le_risco = LabelEncoder()
    y = le_risco.fit_transform(df["risco_credito"].astype(str))
    y_cat = to_categorical(y, num_classes=3)

    print(f"Total de amostras (após sampling): {len(df)}")
    print(f"Recursos numéricos: {num_features_global}")
    print(f"Classes de porte:    {le_porte.classes_}")
    print(f"Classes de risco:    {le_risco.classes_}\n")

    #####################
    # 3) Obtenção de embeddings BERT, com cache
    #####################
    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    bert_model = TFAutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")

    texts = df["justificativa"].astype(str).tolist()

    def load_or_compute_embeddings(texts_list, tokenizer, model):
        if os.path.exists(cache_embeddings_path) and os.path.exists(cache_texts_path):
            with open(cache_texts_path, "rb") as f:
                cached_texts = pickle.load(f)
            if cached_texts == texts_list:
                print("→ Carregando embeddings BERT do cache...")
                return np.load(cache_embeddings_path)
            else:
                print("→ Textos alteraram: recalculando embeddings BERT...")
        else:
            print("→ Cache de embeddings não encontrado: calculando embeddings BERT...")

        X_emb = get_bert_embeddings(texts_list, tokenizer, model)
        np.save(cache_embeddings_path, X_emb)
        with open(cache_texts_path, "wb") as f:
            pickle.dump(texts_list, f)
        print(f"→ Embeddings salvos em: {cache_embeddings_path} e {cache_texts_path}")
        return X_emb

    X_text = load_or_compute_embeddings(texts, tokenizer, bert_model)
    print(f"Dimensão dos embeddings: {X_text.shape}\n")

    #####################
    # 4) Treinamento ou carregamento do modelo (cache de treinamento)
    #####################
    if os.path.exists(model_path):
        print(f"→ Modelo treinado encontrado em '{model_path}'. Carregando modelo...")
        best_model = load_model(model_path)
    else:
        print(
            "→ Nenhum modelo pré-treinado encontrado. Executando Keras Tuner e treinamento..."
        )

        def model_builder(hp):
            return build_model(hp, num_portes, X_num.shape[1], X_text.shape[1])

        tuner = Hyperband(
            model_builder,
            objective="val_accuracy",
            max_epochs=5,
            factor=3,
            directory="keras_tuner_dir",
            project_name="credito_bert_justif",
        )
        tuner.search(
            [porte, X_num, X_text],
            y_cat,
            epochs=5,
            validation_split=0.15,
            batch_size=64,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=3),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
            ],
            verbose=1,
        )

        best_model = tuner.get_best_models(num_models=1)[0]

        (
            X_porte_train,
            X_porte_test,
            X_num_train,
            X_num_test,
            X_text_train,
            X_text_test,
            y_train,
            y_test,
        ) = train_test_split(
            porte, X_num, X_text, y_cat, test_size=0.15, random_state=42
        )

        best_model.fit(
            [X_porte_train, X_num_train, X_text_train],
            y_train,
            epochs=50,
            batch_size=32,
            validation_data=([X_porte_test, X_num_test, X_text_test], y_test),
            callbacks=[
                EarlyStopping(
                    monitor="val_loss", patience=4, restore_best_weights=True
                ),
                ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
            ],
            verbose=1,
        )

        # Salvar modelo treinado em disco
        best_model.save(model_path)
        print(f"→ Modelo treinado salvo em '{model_path}'.\n")

    #####################
    # 5) Avaliação (se o modelo tiver acabado de ser carregado, precisamos criar conjunto de teste para relatório)
    #####################
    # Se o modelo já existia, ainda precisamos gerar X_test para avaliar.
    (
        X_porte_train,
        X_porte_test,
        X_num_train,
        X_num_test,
        X_text_train,
        X_text_test,
        y_train,
        y_test,
    ) = train_test_split(porte, X_num, X_text, y_cat, test_size=0.15, random_state=42)

    pred_probs_test = best_model.predict([X_porte_test, X_num_test, X_text_test])
    pred_classes_test = np.argmax(pred_probs_test, axis=1)
    true_classes_test = np.argmax(y_test, axis=1)
    print("\nRelatório de classificação no conjunto de teste:")
    print(
        classification_report(
            true_classes_test, pred_classes_test, target_names=le_risco.classes_
        )
    )

    #####################
    # 6) Explicabilidade SHAP e geração de relatório PDF
    #####################
    background_data = np.concatenate(
        [X_porte_test.reshape(-1, 1), X_num_test, X_text_test], axis=1
    )
    background_sample = shap.sample(background_data, 100)
    explainer = shap.KernelExplainer(model_predict_wrapper, background_sample)

    test_concat = background_data[0:10]
    shap_values = explainer.shap_values(test_concat)

    feature_names = (
        ["porte_empresa"]
        + num_features_global
        + [f"bert_emb_{i}" for i in range(X_text.shape[1])]
    )

    pdf = PDF()
    idxes = train_test_split(
        np.arange(len(df)), porte, X_num, X_text, test_size=0.15, random_state=42
    )[1]

    for i in range(10):
        probs = best_model.predict(
            [
                X_porte_test[i].reshape(1, 1),
                X_num_test[i].reshape(1, -1),
                X_text_test[i].reshape(1, -1),
            ]
        )[0]
        pred_idx = np.argmax(probs)
        classe_predita = le_risco.classes_[pred_idx]

        df_idx = idxes[i]
        justificativa_original = df.loc[df_idx, "justificativa"]
        porte_str = df.loc[df_idx, "porte_empresa"]
        num_feats_dict = {
            "anos_atividade": df.loc[df_idx, "anos_atividade"],
            "rendimento_anual": df.loc[df_idx, "rendimento_anual"],
            "divida_total": df.loc[df_idx, "divida_total"],
        }
        explicacao_llm = gerar_explicacao_llm(
            justificativa_original, porte_str, num_feats_dict, classe_predita
        )

        pdf.add_prediction(
            empresa_idx=i,
            classe_predita=classe_predita,
            prob=probs,
            justificativa_text=explicacao_llm,
            shap_vals=shap_values[i],
            feature_names=feature_names,
        )

    output_path = "relatorio_risco_credito.pdf"
    pdf.output(output_path)
    print(f"\nRelatório PDF gerado: {output_path}")
