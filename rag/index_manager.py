import logging
from typing import List, Optional

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.schema import BaseNode

from common.decorators import singleton
from rag.storage_context_manager import StorageContextManager

logging.basicConfig(level=logging.INFO)  # Set the desired logging level
logger = logging.getLogger(__name__)


@singleton
class IndexManager:
    """
    This class manages vector index stores
    """

    def __init__(self):
        self.storage_context = StorageContextManager().storage_context

        self.embed_model = Settings.embed_model

        self._base_index = self._create_base_vector_index()

    @property
    def base_index(self) -> VectorStoreIndex:
        """
        Returns the base vector index.
        """
        if self._base_index is None:
            raise ValueError("Base index is not initialized.")
        return self._base_index

    def _create_base_vector_index(self) -> VectorStoreIndex:
        """
        Returns a Base VectorStoreIndex object which contains a storage context
        """
        vector_store_index = VectorStoreIndex(
            nodes=[], storage_context=self.storage_context, embed_model=self.embed_model
        )

        vector_store_index.summary = (
            f"--> Vector Store: {self.storage_context.vector_store.__class__.__name__} \n"
            f"--> Doc Store: {self.storage_context.docstore.__class__.__name__} \n"
            f"--> Index Store: {self.storage_context.index_store.__class__.__name__} \n"
            f"--> Embed Model: {self.embed_model.__class__.__name__}({self.embed_model.model_name}) \n"
        )

        logger.info(f"Created a base vector store index:\n{vector_store_index.summary}")

        return vector_store_index

    def create_vector_index_with_docs(
        self, documents: List[Document], transformations: Optional[List[str]] = None
    ) -> VectorStoreIndex:
        """
        Create a vector index with the given documents. No need to specify transformation to parse documents into nodes.
        """
        vector_store_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
            show_progress=True,
            transformations=transformations,
        )

        logger.info(
            f"Created a vector store index with docs: {vector_store_index.summary}"
        )

        return vector_store_index

    def create_vector_index_with_nodes(
        self, nodes: List[BaseNode], transformations: Optional[List[str]] = None
    ) -> VectorStoreIndex:
        """
        Create a vector index with the given nodes. This is called after the transformations by ingestion pipelines
         and nodes are parsed.
        """
        vector_store_index = VectorStoreIndex(
            nodes=nodes,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
            show_progress=True,
            transformations=transformations,
        )

        logger.info(
            f"Created a vector store index with {len(nodes)} nodes: {vector_store_index.summary}"
        )

        return vector_store_index
