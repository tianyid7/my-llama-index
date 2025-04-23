import logging
from typing import List

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url

from configs.rag_config import (
    DOC_STORE_NAMESPACE,
    EMBED_DIM,
    HYBRID_SEARCH_ENABLED,
    INDEX_STORE_NAMESPACE,
    PGVECTOR_CONN_STR,
    PGVECTOR_TABLE,
    REDIS_HOST,
    REDIS_PORT,
)

logging.basicConfig(level=logging.INFO)  # Set the desired logging level
logger = logging.getLogger(__name__)


class IndexManager:
    """
    This class manages vector index stores
    """

    def __init__(self):
        self.storage_context = self._create_storage_context()

        self.embed_model = Settings.embed_model

        self.base_index: VectorStoreIndex = self._create_base_vector_index()

    def _create_base_vector_index(self) -> VectorStoreIndex:
        """
        Returns a Base VectorStoreIndex object which contains a storage context
        """
        vector_store_index = VectorStoreIndex(
            nodes=[], storage_context=self.storage_context, embed_model=self.embed_model
        )

        logger.info(f"Created a base vector store index: {vector_store_index.summary}")

        return vector_store_index

    def _create_storage_context(self) -> StorageContext:
        """
        Create a storage context with the vector store (pgvector), document store (pgvector), and index store (redis).
        """

        # Create the vector store
        if not PGVECTOR_CONN_STR or not PGVECTOR_TABLE:
            raise ValueError("PGVECTOR_CONN_STR and PGVECTOR_TABLE must be set.")

        url = make_url(PGVECTOR_CONN_STR)
        vector_store = PGVectorStore.from_params(
            database=url.database,
            host=url.host,
            password=url.password,
            port=url.port,
            user=url.username,
            table_name=PGVECTOR_TABLE,
            embed_dim=EMBED_DIM,
            hybrid_search=HYBRID_SEARCH_ENABLED,
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40,
                "hnsw_dist_method": "vector_cosine_ops",
            },
        )

        if not REDIS_PORT or not REDIS_HOST:
            raise ValueError("REDIS_PORT and REDIS_HOST must be set.")

        # Create the document store
        docstore = RedisDocumentStore.from_host_and_port(
            host=REDIS_HOST,
            port=REDIS_PORT,
            namespace=DOC_STORE_NAMESPACE,
        )

        # Create the index store
        index_store = RedisIndexStore.from_host_and_port(
            host=REDIS_HOST, port=REDIS_PORT, namespace=INDEX_STORE_NAMESPACE
        )

        return StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=docstore,
            index_store=index_store,
        )

    def create_vector_index_with_docs(
        self, documents: List[Document]
    ) -> VectorStoreIndex:
        """
        Create a vector index with the given documents.
        """
        vector_store_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
            show_progress=True,
        )

        logger.info(
            f"Created a vector store index with docs: {vector_store_index.summary}"
        )

        return vector_store_index
