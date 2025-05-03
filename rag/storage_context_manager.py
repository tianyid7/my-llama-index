import logging

from llama_index.core import StorageContext
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url

from common.decorators import singleton
from configs.rag_config import (
    DOC_STORE,
    DOC_STORE_NAMESPACE,
    EMBED_DIM,
    GRAPH_STORE,
    HYBRID_SEARCH_ENABLED,
    INDEX_STORE,
    INDEX_STORE_NAMESPACE,
    MEMGRAPH_PASSWORD,
    MEMGRAPH_URL,
    MEMGRAPH_USER,
    PGVECTOR_CONN_STR,
    PGVECTOR_TABLE,
    REDIS_HOST,
    REDIS_PORT,
    VECTOR_DB,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@singleton
class StorageContextManager:
    """
    A manager for managing the storage context.
    """

    def __init__(self):
        self.vector_store = None
        self.docstore = None
        self.index_store = None
        self.property_graph_store = None

        self._storage_context = self._create_storage_context()

    @property
    def storage_context(self) -> StorageContext:
        """
        Returns the storage context.
        """
        if self._storage_context is None:
            raise ValueError("Storage context is not initialized.")
        return self._storage_context

    def _create_storage_context(self) -> StorageContext:
        """
        Create a storage context with the vector store (pgvector), document store (pgvector), and index store (redis).
        """

        # Create the vector store
        if VECTOR_DB == "pgvector":
            if not PGVECTOR_CONN_STR or not PGVECTOR_TABLE:
                raise ValueError("PGVECTOR_CONN_STR and PGVECTOR_TABLE must be set.")

            url = make_url(PGVECTOR_CONN_STR)
            self.vector_store = PGVectorStore.from_params(
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

        # Create the document store
        if DOC_STORE == "redis":
            if not REDIS_PORT or not REDIS_HOST:
                raise ValueError("REDIS_PORT and REDIS_HOST must be set.")

            self.docstore = RedisDocumentStore.from_host_and_port(
                host=REDIS_HOST,
                port=REDIS_PORT,
                namespace=DOC_STORE_NAMESPACE,
            )

        # Create the index store
        if INDEX_STORE == "redis":
            if not REDIS_PORT or not REDIS_HOST:
                raise ValueError("REDIS_PORT and REDIS_HOST must be set.")

            self.index_store = RedisIndexStore.from_host_and_port(
                host=REDIS_HOST, port=REDIS_PORT, namespace=INDEX_STORE_NAMESPACE
            )

        # If no vector store is created, raise an error
        if self.vector_store is None:
            raise ValueError(
                "No vector store created. Please check your configuration 'VECTOR_DB'."
            )

        if GRAPH_STORE == "memgraph":
            from llama_index.graph_stores.memgraph import MemgraphPropertyGraphStore

            self.property_graph_store = MemgraphPropertyGraphStore(
                url=MEMGRAPH_URL,
                username=MEMGRAPH_USER,
                password=MEMGRAPH_PASSWORD,
            )

        return StorageContext.from_defaults(
            vector_store=self.vector_store,
            docstore=self.docstore,
            index_store=self.index_store,
            property_graph_store=self.property_graph_store,
        )
