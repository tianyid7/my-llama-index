from typing import Optional

from llama_index.core import Settings
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.readers.base import BaseReader
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache

from configs.rag_config import (
    CACHE_COLLECTION_NAME,
    DOCSTORE_STRATEGY,
    REDIS_HOST,
    REDIS_PORT,
)
from indexing.node_parsers.node_parser import NodeParser
from rag.index_manager import BASE_VECTOR_INDEX


class IngestionPipelineManager:
    """
    A manager for the ingestion pipeline.
    """

    def __init__(
        self,
        reader: BaseReader,
        node_parser: Optional[NodeParser] = None,
        metadata_extractor: Optional[MetadataExtractor] = None,
    ):
        self.reader = reader

        self.pipeline: IngestionPipeline | None = None

    def create_pipeline(self) -> IngestionPipeline:
        transformations = [Settings.embed_model]

        vector_index = BASE_VECTOR_INDEX

        pipeline = IngestionPipeline(
            transformations=transformations,
            vector_store=vector_index.vector_store,
            docstore=vector_index.docstore,
            cache=IngestionCache(
                cache=RedisCache.from_host_and_port(REDIS_HOST, REDIS_PORT),
                collection=CACHE_COLLECTION_NAME,
            ),
            docstore_strategy=DOCSTORE_STRATEGY,
        )

        self.pipeline = pipeline
        return pipeline

    def run_pipeline(self):
        documents = self.reader.load_data()

        self.pipeline.run(documents=documents, show_progress=True, num_workers=4)
