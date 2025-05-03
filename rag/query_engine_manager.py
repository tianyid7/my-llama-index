import logging
from typing import List

import Stemmer
from llama_index.core import (
    PromptTemplate,
    PropertyGraphIndex,
    Settings,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers import AutoMergingRetriever, QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.tools import QueryEngineTool
from llama_index.postprocessor.presidio import PresidioPIINodePostprocessor
from llama_index.retrievers.bm25 import BM25Retriever

from app.prompts import Prompts
from common.decorators import singleton
from rag.post_processors.node_reranker import CustomLLMRerank
from rag.query_engines.async_extensions import (
    AsyncHyDEQueryTransform,
    AsyncRetrieverQueryEngine,
    AsyncTransformQueryEngine,
)
from rag.retrievers.parent_retriever import ParentRetriever
from rag.retrievers.qa_followup_retriever import QAFollowupRetriever, QARetriever

logging.basicConfig(level=logging.INFO)  # Set the desired logging level
logger = logging.getLogger(__name__)


@singleton
class QueryEngineManager:
    def __init__(
        self,
        base_index: VectorStoreIndex | None = None,
        property_graph_index: PropertyGraphIndex | None = None,
        qa_index: VectorStoreIndex | None = None,
        use_refine: bool = True,
        similarity_top_k: int = 5,
        retrieval_strategy: str = "baseline",
        qa_followup: bool = False,
        hybrid_retrieval: bool = False,
        use_node_rerank: bool = False,
        use_hyde: bool = False,
        use_presidio: bool = False,
    ):
        # Synthesizer hyperparameters
        self.use_refine = use_refine

        # Retriever hyperparameters
        self.similarity_top_k = similarity_top_k
        self.retrieval_strategy = retrieval_strategy
        self.qa_followup = qa_followup
        self.hybrid_retrieval = hybrid_retrieval

        # Vector indexes
        self.base_index = base_index
        self.qa_index = qa_index

        # Property Graph Index
        self.property_graph_index = property_graph_index

        # Query engine hyperparameters
        self.use_node_rerank = use_node_rerank
        self.use_hyde = use_hyde
        self.use_presidio = use_presidio

        self.prompts = Prompts()

        self._query_engine = self._create_query_engine()
        self._property_graph_query_engine = (
            self.property_graph_index.as_query_engine()
            if self.property_graph_index
            else None
        )

        logger.info("Initiated Query Engine Manager")

    @property
    def query_engine(self) -> AsyncRetrieverQueryEngine:
        """
        Returns the query engine.
        """
        if self._query_engine is None:
            raise ValueError("Query engine not initialized.")
        return self._query_engine

    @property
    def property_graph_query_engine(self):
        """
        Returns the property graph query engine.
        """
        if self._property_graph_query_engine is None:
            raise ValueError("Property graph query engine not initialized.")
        return self._property_graph_query_engine

    def _create_retriever(self):
        base_retriever = self.base_index.as_retriever(
            similarity_top_k=self.similarity_top_k
        )

        # Choose between retrieval strategies and configurations.
        if self.retrieval_strategy == "auto_merging":
            retriever = AutoMergingRetriever(
                base_retriever, self.base_index.storage_context, verbose=True
            )
        elif self.retrieval_strategy == "parent":
            retriever = ParentRetriever(
                base_retriever, docstore=self.base_index.docstore
            )
        elif self.retrieval_strategy == "baseline":
            retriever = base_retriever
        else:
            raise ValueError(
                f"Invalid retrieval strategy: {self.retrieval_strategy}. "
                "Valid options are: auto_merging, parent, baseline."
            )

        if self.qa_followup:
            if self.qa_index:
                qa_vector_retriever = self.qa_index.as_retriever(
                    similarity_top_k=self.similarity_top_k
                )
            else:
                raise ValueError("qa_index must be provided when qa_followup is True.")
            qa_retriever = QARetriever(
                qa_vector_retriever=qa_vector_retriever, docstore=self.qa_index.docstore
            )
            retriever = QAFollowupRetriever(
                qa_retriever=qa_retriever, base_retriever=retriever
            )

        if self.hybrid_retrieval:
            bm25_retriever = BM25Retriever.from_defaults(
                docstore=self.base_index.docstore,
                similarity_top_k=self.similarity_top_k,
                stemmer=Stemmer.Stemmer("english"),
                language="english",
            )
            retriever = QueryFusionRetriever(
                [retriever, bm25_retriever],
                similarity_top_k=self.similarity_top_k,
                num_queries=1,  # set this to 1 to disable query generation
                mode=FUSION_MODES.RECIPROCAL_RANK,
                use_async=True,
                verbose=True,
                # query_gen_prompt="...",  # TODO: we could override the query generation prompt here
            )

        return retriever

    def _create_synthesizer(self):
        qa_prompt = PromptTemplate(self.prompts.qa_prompt_tmpl)
        refine_prompt = PromptTemplate(self.prompts.refine_prompt_tmpl)

        if self.use_refine:
            synth = get_response_synthesizer(
                text_qa_template=qa_prompt,
                refine_template=refine_prompt,
                response_mode=ResponseMode.COMPACT,
                use_async=True,
            )
        else:
            synth = get_response_synthesizer(
                text_qa_template=qa_prompt,
                response_mode=ResponseMode.COMPACT,
                use_async=True,
            )

        return synth

    def _create_query_engine(self) -> AsyncRetrieverQueryEngine:
        """
        Creates a llamaindex QueryEngine given a
        VectorStoreIndex and hyperparameters
        """
        # Add Node Postprocessors (if any)
        node_postprocessors = self._add_node_postprocessors()

        query_engine = AsyncRetrieverQueryEngine.from_args(
            self._create_retriever(),
            response_synthesizer=self._create_synthesizer(),
            node_postprocessors=node_postprocessors,
        )

        # Add Query Transformation (if any)
        transform_query_engine = self._add_query_transformations(query_engine)

        return transform_query_engine or query_engine

    def _add_node_postprocessors(self) -> List[BaseNodePostprocessor]:
        node_postprocessors = []
        if self.use_node_rerank:
            reranker_llm = Settings.llm  # TODO: use a different LLM for reranking
            choice_select_prompt = PromptTemplate(
                self.prompts.choice_select_prompt_tmpl
            )
            llm_reranker = CustomLLMRerank(
                choice_batch_size=10,
                top_n=5,
                choice_select_prompt=choice_select_prompt,
                llm=reranker_llm,
            )
            node_postprocessors.append(llm_reranker)

        if self.use_presidio:
            presidio_analyzer = PresidioPIINodePostprocessor()
            node_postprocessors.append(presidio_analyzer)

        return node_postprocessors

    def _add_query_transformations(
        self, query_engine
    ) -> AsyncTransformQueryEngine | None:
        """Adds query transformations to the query engine.
        see https://docs.llamaindex.ai/en/stable/examples/query_transformations/query_transform_cookbook/
        """
        if self.use_hyde:
            hyde_prompt = PromptTemplate(self.prompts.hyde_prompt_tmpl)
            hyde = AsyncHyDEQueryTransform(
                include_original=True, hyde_prompt=hyde_prompt
            )
            transform_query_engine = AsyncTransformQueryEngine(
                query_engine=query_engine,
                query_transform=hyde,
            )
            return transform_query_engine
        else:
            return

    def get_query_engine_tool(
        self, name, description, index_type: str = "vector"
    ) -> QueryEngineTool:
        """
        Returns a llamaindex QueryEngineTool
        """
        if index_type == "graph":
            query_engine = self.property_graph_query_engine
        elif index_type == "vector":
            query_engine = self.query_engine
        else:
            raise ValueError(
                f"Invalid index type: {index_type}. Supported index types are: graph, vector."
            )

        if query_engine is None:
            raise ValueError(
                "Query engine not created. Call _create_query_engine first."
            )

        query_engine_tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name=name,
            description=description,
        )
        return query_engine_tool
