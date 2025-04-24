import logging
from typing import List

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.schema import BaseNode

from rag.storage_context_manager import StorageContextManager

logging.basicConfig(level=logging.INFO)  # Set the desired logging level
logger = logging.getLogger(__name__)


class IndexManager:
    """
    This class manages vector index stores
    """

    def __init__(self):
        self.storage_context = StorageContextManager().create_storage_context()

        self.embed_model = Settings.embed_model

    def create_base_vector_index(self) -> VectorStoreIndex:
        """
        Returns a Base VectorStoreIndex object which contains a storage context
        """
        vector_store_index = VectorStoreIndex(
            nodes=[], storage_context=self.storage_context, embed_model=self.embed_model
        )

        logger.info(f"Created a base vector store index: {vector_store_index.summary}")

        return vector_store_index

    def create_vector_index_with_docs(
        self, documents: List[Document]
    ) -> VectorStoreIndex:
        """
        Create a vector index with the given documents. No need to specify transformation to parse documents into nodes.
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

    def create_vector_index_with_nodes(self, nodes: List[BaseNode]) -> VectorStoreIndex:
        """
        Create a vector index with the given nodes. This is called after the transformations by ingestion pipelines
         and nodes are parsed.
        """
        vector_store_index = VectorStoreIndex(
            nodes=nodes,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
            show_progress=True,
        )

        logger.info(
            f"Created a vector store index with {len(nodes)} nodes: {vector_store_index.summary}"
        )

        return vector_store_index


# Create a singleton base index
BASE_VECTOR_INDEX = IndexManager().create_base_vector_index()
