import os
import pickle
from typing import List

import numpy as np
from transformers import AutoTokenizer, TFAutoModel

# Constantes de configuração
MAX_LEN = 64
BATCH_SIZE = 32
CACHE_EMBEDDINGS_FILENAME = "cache_X_text.npy"
CACHE_TEXTS_FILENAME = "cache_texts.pkl"


def get_bert_embeddings(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: TFAutoModel,
    max_len: int = MAX_LEN,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Calcula embeddings (vetor [CLS]) para uma lista de textos 
    usando modelo BERT.

    Retorna um ndarray de shape (len(texts), hidden_size).
    """
    embeddings_list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="tf",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        outputs = model(inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings_list.append(cls_embeddings)
    return np.vstack(embeddings_list)


def load_or_compute_embeddings(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: TFAutoModel,
    cache_dir: str = None,
) -> np.ndarray:
    """
    Carrega embeddings de cache se disponível e inalterado,
    caso contrário, recalcula e armazena em disco.

    - texts: lista de strings a serem embedadas
    - tokenizer, model: instâncias do HuggingFace
    - cache_dir: diretório onde ficam os arquivos de cache

    Retorna um ndarray com embeddings salvo/recuperado.
    """
    cache_emb_path = (
        os.path.join(cache_dir, CACHE_EMBEDDINGS_FILENAME)
        if cache_dir
        else CACHE_EMBEDDINGS_FILENAME
    )
    cache_texts_path = (
        os.path.join(cache_dir, CACHE_TEXTS_FILENAME)
        if cache_dir
        else CACHE_TEXTS_FILENAME
    )

    # Se cache existe, verificar se os textos são iguais
    if os.path.exists(cache_emb_path) and os.path.exists(cache_texts_path):
        with open(cache_texts_path, "rb") as f:
            cached_texts = pickle.load(f)
        if cached_texts == texts:
            print("Carregando embeddings do cache...")
            return np.load(cache_emb_path)
        print("Textos alterados: recalculando embeddings...")
    else:
        print("Cache de embeddings não encontrado: calculando embeddings...")

    # Gerar embeddings
    emb = get_bert_embeddings(texts, tokenizer, model)

    # Salvar no cache
    np.save(cache_emb_path, emb)
    with open(cache_texts_path, "wb") as f:
        pickle.dump(texts, f)
    print(f"Embeddings salvos em: {cache_emb_path} e {cache_texts_path}")

    return emb
