"""Main state management class for indices and prompts for
experimentation UI"""

import logging

from llama_index.core import (
    PromptTemplate,
    Settings,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.retrievers import AutoMergingRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url

from app.prompts import Prompts
from rag.async_extensions import (
    AsyncHyDEQueryTransform,
    AsyncRetrieverQueryEngine,
    AsyncTransformQueryEngine,
)
from rag.node_reranker import CustomLLMRerank
from rag.retrievers.parent_retriever import ParentRetriever
from rag.retrievers.qa_followup_retriever import QAFollowupRetriever, QARetriever

logging.basicConfig(level=logging.INFO)  # Set the desired logging level
logger = logging.getLogger(__name__)


class IndexManager:
    """
    This class manages state for vector indexes,
    docstores, query engines and chat engines
    across the app's lifecycle (e.g. through UI manipulations).
    The index_manager (instantiated) will be injected into all API calls
    that need to access its state or manipulate its state.
    This includes:
    - Switching out vector indices or docstores
    - Changing retrieval parameters (e.g. temperature, llm model, etc.)
    """

    def __init__(
        self,
        pgvector_conn: str,
        redis_host: str,
    ):
        # Stores
        self.pgvector_conn = pgvector_conn
        self.redis_host = redis_host

        self.embed_model = Settings.embed_model

        self.base_index = self.get_vector_index(
            pgvector_conn=self.pgvector_conn,
            redis_host=self.redis_host,
        )

        self.query_engine = None

    def get_vector_index(
        self,
        pgvector_conn: str,
        redis_host: str,
        redis_port: int = 6379,
    ) -> VectorStoreIndex:
        """
        Returns a llamaindex VectorStoreIndex object which contains a storage context
        """
        # Create the vector store
        url = make_url(pgvector_conn)
        vector_store = PGVectorStore.from_params(
            database=url.database,
            host=url.host,
            password=url.password,
            port=url.port,
            user=url.username,
            table_name="paul_graham_essay",
            embed_dim=1536,  # openai embedding dimension
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40,
                "hnsw_dist_method": "vector_cosine_ops",
            },
        )

        # Create the document store TODO: doc_store and index_store might better be the same with vector_store
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

    def get_current_index_info(self) -> dict:
        """Return the indices currently being used"""
        return {
            "vector_store": {
                "type": "PGVectorStore",
                "connection": self.pgvector_conn,
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

    def set_current_index(
        self, pgvector_conn: str, redis_host: str, redis_port: int = 6379
    ) -> None:
        """Set the current indices to be used for the RAG"""
        self.base_index = self.get_vector_index(
            pgvector_conn,
            redis_host,
            redis_port,
        )
