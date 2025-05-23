import os

# This file contains the configuration for the RAG (Retrieval-Augmented Generation) system.
# It includes settings for the vector store, document store, and index store, etc.

VECTOR_DB = os.getenv("VECTOR_DB", "pgvector")
DOC_STORE = os.getenv("DOC_STORE", "redis")
INDEX_STORE = os.getenv("INDEX_STORE", "redis")
GRAPH_STORE = os.getenv("GRAPH_STORE", "memgraph")

# ********* pgvector configs ********* #
PGVECTOR_CONN_STR = os.getenv(
    "PGVECTOR_CONN_STR", "postgresql://postgres:postgres@localhost:5431/vectordb"
)
PGVECTOR_TABLE = os.getenv("PGVECTOR_TABLE", "index")
HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "false") == "true"
EMBED_DIM = os.getenv("EMBED_DIM", 1536)

# ********* redis configs ********* #
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
# namespace if using Redis as a document store
DOC_STORE_NAMESPACE = os.getenv("DOC_STORE_NAMESPACE", "llama_index_doc")
# namespace if using Redis as an index store
INDEX_STORE_NAMESPACE = os.getenv("INDEX_STORE_NAMESPACE", "llama_index_index")
# collection name if using Redis as cache in ingestion pipeline
CACHE_COLLECTION_NAME = os.getenv("CACHE_COLLECTION_NAME", "llama_index_indexing_cache")

# ********* memgraph configs ********* #
MEMGRAPH_URL = os.getenv("MEMGRAPH_URL", "bolt://localhost:7687")
MEMGRAPH_USER = os.getenv("MEMGRAPH_USER", "memgraph")
MEMGRAPH_PASSWORD = os.getenv("MEMGRAPH_PASSWORD", "memgraph")

# ********* ingestion pipeline configs ********* #
# docstore strategy for the ingestion pipeline, can only be "duplicates_only", "upserts", and "upserts_and_delete"
# see https://docs.llamaindex.ai/en/stable/api_reference/ingestion/#llama_index.core.ingestion.pipeline.DocstoreStrategy
DOCSTORE_STRATEGY = os.getenv("DOCSTORE_STRATEGY", "upserts")
