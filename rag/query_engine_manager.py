import logging

import Stemmer
from llama_index.core import (
    PromptTemplate,
    Settings,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.retrievers import AutoMergingRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

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


class QueryEngineManager:
    query_engine: AsyncRetrieverQueryEngine | None = None

    def create_query_engine(
        self,
        prompts: Prompts,
        similarity_top_k: int = 5,
        retrieval_strategy: str = "auto_merging",
        base_index: VectorStoreIndex | None = None,
        qa_index: VectorStoreIndex | None = None,
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

        base_retriever = base_index.as_retriever(similarity_top_k=similarity_top_k)

        if qa_index:
            qa_vector_retriever = qa_index.as_retriever(
                similarity_top_k=similarity_top_k
            )
        else:
            qa_vector_retriever = None
        query_engine = None  # Default initialization

        # Choose between retrieval strategies and configurations.
        if retrieval_strategy == "auto_merging":
            logger.info(base_index.storage_context.docstore)
            retriever = AutoMergingRetriever(
                base_retriever, base_index.storage_context, verbose=True
            )
        elif retrieval_strategy == "parent":
            retriever = ParentRetriever(base_retriever, docstore=base_index.docstore)
        elif retrieval_strategy == "baseline":
            retriever = base_retriever

        if qa_followup:
            qa_retriever = QARetriever(
                qa_vector_retriever=qa_vector_retriever, docstore=qa_index.docstore
            )
            retriever = QAFollowupRetriever(
                qa_retriever=qa_retriever, base_retriever=retriever
            )

        if hybrid_retrieval:
            bm25_retriever = BM25Retriever.from_defaults(
                docstore=base_index.docstore,
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
