import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Carrega variáveis de ambiente do .env (se existir)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")  # opcional para v3
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "justificativas-index")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "gcp")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-central1")


def get_or_create_index(dimension: int = 768, metric: str = "cosine"):
    """
    Inicializa o cliente Pinecone e cria (ou conecta) ao índice.
    Usa GCP/us-central1 por padrão, compatível com o plano gratuito.

    dimension: número de dimensões do vetor (e.g. 768 para BERT).
    metric: "cosine", "dotproduct" ou "euclidean".
    """
    if not PINECONE_API_KEY:
        raise ValueError("A variável de ambiente PINECONE_API_KEY não está definida.")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = pc.list_indexes().names()
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(
            f"Criando índice '{PINECONE_INDEX_NAME}' (dim={dimension}, metric={metric})..."
        )
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    else:
        print(f"Índice '{PINECONE_INDEX_NAME}' já existe.")

    return pc.Index(PINECONE_INDEX_NAME)


# ---------------------------------------------------------------
# ✅ Função 1: Buscar justificativas similares via Pinecone
def search_similar_justifications(index, query_embedding, top_k=5):
    """
    Realiza busca vetorial no índice Pinecone e retorna os top_k mais similares.

    index: instância do índice Pinecone.
    query_embedding: vetor de embedding (ex: BERT).
    top_k: número de resultados mais similares a retornar.
    """
    query_vector = query_embedding.tolist()
    result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return result


# ---------------------------------------------------------------
# ✅ Função 2: RAG – Gerar explicação com base em justificativas similares
def gerar_explicacao_rag(query_justificativa, similares, modelo_llm):
    """
    Usa as justificativas similares recuperadas para gerar explicação contextualizada via LLM.
    """
    contexto = "\n".join([f"- {s}" for s in similares])
    prompt = f"""
Justificativa do cliente:
{query_justificativa}

Justificativas similares:
{contexto}

Com base nessas informações, explique o risco de crédito desse cliente de forma clara e 
fundamentada.
"""
    return modelo_llm(prompt)


# ---------------------------------------------------------------
# ✅ Função 3: Visualizar conteúdo do índice Pinecone (com base em cache de IDs)
def view_index_contents(index, ids=None):
    """
    Busca vetores e metadados diretamente no índice a partir de uma lista de IDs.
    Pinecone não permite listar todos os IDs diretamente.
    """
    if not ids:
        print(
            "Você precisa passar uma lista de IDs (use cache local ou salve ao inserir)."
        )
        return
    result = index.fetch(ids=ids)
    return result
