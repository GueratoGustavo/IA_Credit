import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Carrega variáveis de ambiente do .env (se existir)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv(
    "PINECONE_ENVIRONMENT"
)  # ex: “gcp-starter” (opcional para v3)
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

    # Inicializa o cliente Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # (Opcional) Se precisar enviar environment, você pode instanciar assim:
    # pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

    # Obtém lista de índices existentes
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
