import logging

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url

logging.basicConfig(
    level=logging.INFO
)  # Set the desired logging level #TODO Make this configurable
logger = logging.getLogger(__name__)


class IndexManager:
    """
    This class manages index stores for
    - vector (pgvector)
    - indexes (redis)
    - docs (redis)
    """

    def __init__(
        self,
        pgvector_conn: str,
        pgvector_table: str,
        redis_host: str,
    ):
        # Stores
        self.pgvector_conn = pgvector_conn
        self.pgvector_table = pgvector_table
        self.redis_host = redis_host

        self.embed_model = Settings.embed_model

        self.base_index = self._create_vector_index(
            pgvector_conn=self.pgvector_conn,
            pgvector_table=self.pgvector_table,
            redis_host=self.redis_host,
        )

    def get_current_index_info(self) -> dict:
        """Return the indices currently being used"""
        return {
            "vector_store": {
                "type": "PGVectorStore",
                "connection": self.pgvector_conn,
                "table": self.pgvector_table,
            },
            "docstore": {
                "type": "RedisDocumentStore",
                "host": self.redis_host,
                "port": 6379,
            },
            "index_store": {
                "type": "RedisIndexStore",
                "host": self.redis_host,
                "port": 6379,
            },
            "embed_model": self.embed_model.model_name,
            "summary": self.base_index.summary,
        }

    def _create_vector_index(
        self,
        pgvector_conn: str,
        pgvector_table: str,
        redis_host: str,
        redis_port: int = 6379,
    ) -> VectorStoreIndex:
        """
        Returns a VectorStoreIndex object which contains a storage context
        """
        # Create the vector store
        url = make_url(pgvector_conn)
        vector_store = PGVectorStore.from_params(
            database=url.database,
            host=url.host,
            password=url.password,
            port=url.port,
            user=url.username,
            table_name=pgvector_table,
            embed_dim=1536,
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40,
                "hnsw_dist_method": "vector_cosine_ops",
            },
        )

        # Create the document store
        docstore = RedisDocumentStore.from_host_and_port(
            host=redis_host,
            port=redis_port,
            namespace="llama_index_doc",  # TODO: make this configurable
        )

        # Create the index store
        index_store = RedisIndexStore.from_host_and_port(
            host=redis_host, port=redis_port, namespace="llama_index"
        )

        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, docstore=docstore, index_store=index_store
        )

        # Create and return the index
        vector_store_index = VectorStoreIndex(
            nodes=[], storage_context=storage_context, embed_model=self.embed_model
        )
        return vector_store_index
