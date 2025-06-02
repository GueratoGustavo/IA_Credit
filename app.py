import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import requests  # Added for potential xAI API calls
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
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


def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    df["anos_atividade"] = df["anos_atividade"].fillna(0).astype(int)
    df["rendimento_anual"] = df["rendimento_anual"].fillna(0).clip(lower=0)
    df["divida_total"] = df["divida_total"].fillna(0)
    df["justificativa"] = df["justificativa"].fillna("sem justificativa")

    num_features = ["anos_atividade", "rendimento_anual", "divida_total"]
    X_num = StandardScaler().fit_transform(df[num_features].astype(np.float32))

    le_porte = LabelEncoder()
    porte = le_porte.fit_transform(df["porte_empresa"].astype(str))
    num_portes = len(le_porte.classes_)

    le_risco = LabelEncoder()
    y = le_risco.fit_transform(df["risco_credito"].astype(str))
    y_cat = to_categorical(y, num_classes=3)

    return df, X_num, porte, num_portes, y, y_cat, le_porte, le_risco, num_features


def get_bert_embeddings(texts, tokenizer, model, max_len=64, batch_size=32):
    embeddings = []
    texts = [
        str(t) if not isinstance(t, str) else t
        for t in (texts.tolist() if isinstance(texts, pd.Series) else texts)
    ]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
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


def model_predict(data):
    porte_part = data[:, 0].astype(int).reshape(-1, 1)
    num_part = data[:, 1:4]
    text_part = data[:, 4:]
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
        self, empresa_idx, classe_predita, prob, justificativa, shap_vals, feature_names
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
        self.multi_cell(0, 10, justificativa)

        plt.figure(figsize=(6, 3))
        shap.summary_plot(shap_vals, feature_names=feature_names, show=False)
        plt.tight_layout()
        imagem_path = f"shap_plot_{empresa_idx}.png"
        plt.savefig(imagem_path)
        plt.close()

        self.image(imagem_path, x=10, w=180)
        os.remove(imagem_path)


def generate_grok3_response(
    justificativa, porte, num_features_dict, classe_predita, max_tokens=250, temp=0.7
):
    """
        Simulates Grok 3 response generation. Replace with xAI API call for production.
        To use the xAI API:
        1. Obtain an API key from https://x.ai/api
        2. Send a POST request to the appropriate endpoint (e.g., https://api.x.ai/grok3)
        3. Include prompt, max_tokens, and temperature in the payload
        Example API call:

    python
        headers = {"Authorization": f"Bearer YOUR_API_KEY"}
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temp
        }
        response = requests.post("https://api.x.ai/grok3", json=payload, headers=headers)
        return response.json()["choices"][0]["text"].strip()

    """
    prompt = f"""
Você é um analista de crédito. Analise os dados da empresa a seguir e explique de forma clara por que ela foi classificada como '{classe_predita}'.

Porte da empresa: {porte}
Rendimento anual: {num_features_dict['rendimento_anual']:.2f}
Anos de atividade: {num_features_dict['anos_atividade']}
Dívida total: {num_features_dict['divida_total']:.2f}
Justificativa fornecida: "{justificativa}"

Explique a classificação de risco atribuída à empresa (classe: {classe_predita}) com base nos dados acima, usando linguagem clara e profissional.
"""
    # Mock response for prototyping
    mock_response = f"""
A empresa foi classificada como '{classe_predita}' com base nos seguintes fatores:
- **Porte da empresa ({porte})**: Empresas de porte {porte} geralmente apresentam características que influenciam o risco de crédito.
- **Rendimento anual ({num_features_dict['rendimento_anual']:.2f})**: Um rendimento de {num_features_dict['rendimento_anual']:.2f} sugere uma capacidade financeira que impacta a classificação.
- **Anos de atividade ({num_features_dict['anos_atividade']})**: {num_features_dict['anos_atividade']} anos de atividade indicam estabilidade ou falta dela.
- **Dívida total ({num_features_dict['divida_total']:.2f})**: Uma dívida de {num_features_dict['divida_total']:.2f} reflete o nível de endividamento.
- **Justificativa fornecida**: "{justificativa}" foi analisada, indicando fatores adicionais que corroboram a classificação.

Com base nesses dados, a classificação '{classe_predita}' é justificada devido a uma combinação de fatores financeiros e operacionais.
"""
    # Truncate to simulate max_tokens
    words = mock_response.split()
    return " ".join(words[:max_tokens]).strip()


def gerar_explicacao_llm(justificativa, porte, num_features_dict, classe_predita):
    output = generate_grok3_response(
        justificativa,
        porte,
        num_features_dict,
        classe_predita,
        max_tokens=250,
        temp=0.7,
    )
    return output.strip()


if __name__ == "__main__":
    print("Usando Grok 3 para explicações (simulado ou via API)...")
    df, X_num, porte, num_portes, y, y_cat, le_porte, le_risco, num_features = (
        load_and_prepare_data("empresas_credito_200k_justificativa.csv")
    )

    num_samples = 1000
    sample_indices = df.sample(n=num_samples, random_state=42).index
    df_sample = df.loc[sample_indices].reset_index(drop=True)
    X_num_sample = X_num[sample_indices]
    porte_sample = porte[sample_indices]
    y_sample = y[sample_indices]
    y_cat_sample = y_cat[sample_indices]

    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    bert_model = TFAutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")
    bert_model.trainable = False

    print(f"Extraindo embeddings BERT para {num_samples} amostras... (rápido)")
    X_text_sample = get_bert_embeddings(
        df_sample["justificativa"], tokenizer, bert_model, max_len=64, batch_size=64
    )
    X_text_sample = StandardScaler().fit_transform(X_text_sample)

    (
        porte_train,
        porte_test,
        X_num_train,
        X_num_test,
        X_text_train,
        X_text_test,
        y_train,
        y_test,
    ) = train_test_split(
        porte_sample,
        X_num_sample,
        X_text_sample,
        y_cat_sample,
        test_size=0.2,
        random_state=42,
        stratify=y_sample,
    )

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_sample), y=y_sample
    )
    class_weights_dict = dict(enumerate(class_weights))

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-6),
    ]

    tuner = Hyperband(
        lambda hp: build_model(
            hp, num_portes, len(num_features), X_text_train.shape[1]
        ),
        objective="val_accuracy",
        max_epochs=20,
        factor=3,
        seed=42,
        directory="tuner_dir",
        project_name="credit_risk_bert_1000",
    )
    tuner.search(
        [porte_train, X_num_train, X_text_train],
        y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.2,
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=1,
    )

    global best_model
    best_model = tuner.get_best_models(num_models=1)[0]
    test_loss, test_acc = best_model.evaluate(
        [porte_test, X_num_test, X_text_test], y_test
    )
    print(f"Teste Loss: {test_loss:.4f} | Teste Accuracy: {test_acc:.4f}")

    y_pred_prob = best_model.predict([porte_test, X_num_test, X_text_test])
    y_pred = y_pred_prob.argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    print(classification_report(y_true, y_pred, target_names=le_risco.classes_))

    # -------- EXPLICABILIDADE SHAP --------
    print("Calculando explicabilidade SHAP...")
    feature_names = (
        ["porte_empresa"]
        + num_features
        + [f"bert_feat_{i}" for i in range(X_text_train.shape[1])]
    )

    X_shap = np.hstack((porte_test.reshape(-1, 1), X_num_test, X_text_test))
    explainer = shap.Explainer(model_predict, X_shap[:100], max_evals=2000)
    shap_values = explainer(X_shap[:10])

    pdf = PDF()
    for i in range(10):
        classe_predita = le_risco.inverse_transform([y_pred[i]])[0]
        justificativa = df_sample["justificativa"].iloc[i]
        num_dict = {
            "anos_atividade": X_num_test[i][0],
            "rendimento_anual": X_num_test[i][1],
            "divida_total": X_num_test[i][2],
        }
        porte_str = le_porte.inverse_transform([porte_test[i]])[0]

        explicacao_llm = gerar_explicacao_llm(
            justificativa=justificativa,
            porte=porte_str,
            num_features_dict=num_dict,
            classe_predita=classe_predita,
        )

        pdf.add_prediction(
            empresa_idx=i,
            classe_predita=classe_predita,
            prob=y_pred_prob[i],
            justificativa=justificativa + "\n\nExplicação da LLM:\n" + explicacao_llm,
            shap_vals=shap_values[i].values,
            feature_names=feature_names,
        )

    pdf.output("relatorio_risco_empresas.pdf")
    print("Relatório PDF gerado com sucesso.")
