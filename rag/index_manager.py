"""Main state management class for indices and prompts for
experimentation UI"""

import logging

import Stemmer
from app.prompts import Prompts
from rag.async_extensions import (
    AsyncHyDEQueryTransform,
    AsyncRetrieverQueryEngine,
    AsyncTransformQueryEngine,
)
from rag.node_reranker import CustomLLMRerank
from rag.retrievers.parent_retriever import ParentRetriever
from rag.retrievers.qa_followup_retriever import QAFollowupRetriever, QARetriever
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url
from llama_index.core import (
    PromptTemplate,
    Settings,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
)

from llama_index.core.retrievers import AutoMergingRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

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
            host=redis_host, port=redis_port, namespace="llama_index_doc"  # TODO: make this configurable
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
            self,
            pgvector_conn: str,
            redis_host: str,
            redis_port: int = 6379
    ) -> None:
        """Set the current indices to be used for the RAG"""
        self.base_index = self.get_vector_index(
            pgvector_conn,
            redis_host,
            redis_port,
        )

    def get_query_engine(
            self,
            prompts: Prompts,
            similarity_top_k: int = 5,
            retrieval_strategy: str = "auto_merging",
            use_hyde: bool = True,
            use_refine: bool = True,
            use_node_rerank: bool = False,
            qa_followup: bool = True,
            hybrid_retrieval: bool = True,
    ) -> AsyncRetrieverQueryEngine:
        """
        Creates a llamaindex QueryEngine given a
        VectorStoreIndex and hyperparameters
        """
        llm = Settings.llm

        qa_prompt = PromptTemplate(prompts.qa_prompt_tmpl)
        refine_prompt = PromptTemplate(prompts.refine_prompt_tmpl)

        if use_refine:
            synth = get_response_synthesizer(
                text_qa_template=qa_prompt,
                refine_template=refine_prompt,
                response_mode="compact",
                use_async=True,
            )
        else:
            synth = get_response_synthesizer(
                text_qa_template=qa_prompt, response_mode="compact", use_async=True
            )

        base_retriever = self.base_index.as_retriever(similarity_top_k=similarity_top_k)
        if self.qa_index:
            qa_vector_retriever = self.qa_index.as_retriever(
                similarity_top_k=similarity_top_k
            )
        else:
            qa_vector_retriever = None
        query_engine = None  # Default initialization

        # Choose between retrieval strategies and configurations.
        if retrieval_strategy == "auto_merging":
            logger.info(self.base_index.storage_context.docstore)
            retriever = AutoMergingRetriever(
                base_retriever, self.base_index.storage_context, verbose=True
            )
        elif retrieval_strategy == "parent":
            retriever = ParentRetriever(
                base_retriever, docstore=self.base_index.docstore
            )
        elif retrieval_strategy == "baseline":
            retriever = base_retriever

        if qa_followup:
            qa_retriever = QARetriever(
                qa_vector_retriever=qa_vector_retriever, docstore=self.qa_index.docstore
            )
            retriever = QAFollowupRetriever(
                qa_retriever=qa_retriever, base_retriever=retriever
            )

        if hybrid_retrieval:
            bm25_retriever = BM25Retriever.from_defaults(
                docstore=self.base_index.docstore,
                similarity_top_k=similarity_top_k,
                stemmer=Stemmer.Stemmer("english"),
                language="english",
            )
            retriever = QueryFusionRetriever(
                [retriever, bm25_retriever],
                similarity_top_k=similarity_top_k,
                num_queries=1,  # set this to 1 to disable query generation
                mode="reciprocal_rerank",
                use_async=True,
                verbose=True,
                # query_gen_prompt="...",  # we could override the
                # query generation prompt here
            )

        if use_node_rerank:
            reranker_llm = llm  # TODO: use a different LLM for reranking
            choice_select_prompt = PromptTemplate(prompts.choice_select_prompt_tmpl)
            llm_reranker = CustomLLMRerank(
                choice_batch_size=10,
                top_n=5,
                choice_select_prompt=choice_select_prompt,
                llm=reranker_llm,
            )
        else:
            llm_reranker = None

        query_engine = AsyncRetrieverQueryEngine.from_args(
            retriever,
            response_synthesizer=synth,
            node_postprocessors=[llm_reranker] if llm_reranker else None,
        )

        if use_hyde:
            hyde_prompt = PromptTemplate(prompts.hyde_prompt_tmpl)
            hyde = AsyncHyDEQueryTransform(
                include_original=True, hyde_prompt=hyde_prompt
            )
            query_engine = AsyncTransformQueryEngine(
                query_engine=query_engine, query_transform=hyde
            )

        self.query_engine = query_engine

        return query_engine
