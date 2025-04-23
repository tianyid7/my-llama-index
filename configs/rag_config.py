import os

# This file contains the configuration for the RAG (Retrieval-Augmented Generation) system.
# It includes settings for the vector store, document store, and index store, etc.

# pgvector configs
PGVECTOR_CONN_STR = os.getenv(
    "PGVECTOR_CONN_STR", "postgresql://postgres:postgres@localhost:5431/vectordb"
)
PGVECTOR_TABLE = os.getenv("PGVECTOR_TABLE", "document")
HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "false") == "true"
EMBED_DIM = os.getenv("EMBED_DIM", 1536)

# redis configs
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
DOC_STORE_NAMESPACE = os.getenv("DOC_STORE_NAMESPACE", "llama_index_doc")
INDEX_STORE_NAMESPACE = os.getenv("INDEX_STORE_NAMESPACE", "llama_index")
