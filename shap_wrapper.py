import numpy as np
import shap
from typing import Optional

# O best_model será atribuído externamente (ex: main.py)
best_model: Optional[object] = None

# Lista das features numéricas usadas (para organização e referência)
num_features_global = ["anos_atividade", "rendimento_anual", "divida_total"]


def model_predict_wrapper(data_numpy: np.ndarray) -> np.ndarray:
    """
    Wrapper para o modelo Keras que recebe um array numpy concatenado e retorna
    previsões de probabilidade do modelo.

    Args:
        data_numpy: np.ndarray de shape (n_samples, 1 + len(num_features_global) + embedding_dim),
                    onde:
                    - coluna 0 é a variável categórica 'porte_empresa' codificada como int,
                    - colunas 1 até 1+len(num_features_global) são as variáveis numéricas,
                    - colunas restantes são os embeddings de texto.

    Returns:
        np.ndarray com as probabilidades preditas pelo modelo para cada classe.
    """
    if best_model is None:
        raise ValueError("O modelo não foi carregado para o SHAP wrapper.")

    porte_part = data_numpy[:, 0].astype(int).reshape(-1, 1)
    num_part = data_numpy[:, 1 : 1 + len(num_features_global)]
    text_part = data_numpy[:, 1 + len(num_features_global) :]

    # Chama o modelo keras com a entrada esperada: [porte, num, texto]
    return best_model.predict([porte_part, num_part, text_part])


def compute_shap_values(
    porte_features: np.ndarray,
    num_features: np.ndarray,
    text_embeddings: np.ndarray,
    background_size: int = 100,
    nsamples: int = 100,
) -> list:
    """
    Gera valores SHAP para um subconjunto de dados.

    Args:
        porte_features: np.ndarray shape (n_samples, 1) - variável categórica porte_empresa codificada.
        num_features: np.ndarray shape (n_samples, len(num_features_global)) - variáveis numéricas.
        text_embeddings: np.ndarray shape (n_samples, embedding_dim) - embeddings do texto.
        background_size: int - número de amostras do background para KernelExplainer.
        nsamples: int - número de amostras para cálculo de SHAP.

    Returns:
        list de np.ndarray contendo os valores SHAP para cada classe (multi-output).
    """
    # Concatena as features em uma única matriz para o explainer
    data_concat = np.concatenate(
        [porte_features.reshape(-1, 1), num_features, text_embeddings], axis=1
    )

    # Ajusta tamanho do background se maior que a quantidade de dados
    if background_size > data_concat.shape[0]:
        background_size = data_concat.shape[0]

    # Seleciona aleatoriamente amostras para o background
    background_inds = np.random.choice(
        data_concat.shape[0], background_size, replace=False
    )
    background_data = data_concat[background_inds]

    # Inicializa o KernelExplainer do SHAP
    explainer = shap.KernelExplainer(model_predict_wrapper, background_data)

    # Calcula SHAP values para as primeiras nsamples amostras
    explain_data = data_concat[:nsamples]

    shap_values = explainer.shap_values(explain_data)

    return shap_values
