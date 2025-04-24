import logging
from typing import List, Optional, Sequence

from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache

from configs.rag_config import (
    CACHE_COLLECTION_NAME,
    DOCSTORE_STRATEGY,
    REDIS_HOST,
    REDIS_PORT,
)
from indexing.transformations.metadata_extractors.metadata_extractor import (
    MetadataExtractor,
)
from indexing.transformations.node_parsers.node_parser import NodeParser
from indexing.transformations.provided_transformations import PROVIDED_TRANSFORMATIONS
from rag.index_manager import IndexManager

logging.basicConfig(level=logging.INFO)  # Set the desired logging level
logger = logging.getLogger(__name__)


def prepare_transformations(
    transformations: Optional[List[NodeParser | MetadataExtractor | str]] = None,
    **kwargs,
) -> List[TransformComponent]:
    """
    Prepare the transformations for the pipeline.
    """
    if not transformations:
        return []

    result = []
    for t in transformations:
        transformation = getattr(PROVIDED_TRANSFORMATIONS, t)(kwargs)
        result.append(transformation)

    return result


class IngestionPipelineBuilder:
    """
    A manager for the ingestion pipeline.
    """

    def __init__(
        self,
        reader: BaseReader,
        transformations: List[NodeParser | MetadataExtractor | str],
        **kwargs,
    ):
        self.reader = reader

        if not transformations:
            raise ValueError("Transformations must be provided.")

        self.transformations = prepare_transformations(transformations, **kwargs)

        self.pipeline: IngestionPipeline | None = None

    def create_pipeline(self) -> IngestionPipeline:
        vector_index = IndexManager().base_index

        pipeline = IngestionPipeline(
            transformations=self.transformations,
            docstore=vector_index.docstore,
            cache=IngestionCache(
                cache=RedisCache.from_host_and_port(REDIS_HOST, REDIS_PORT),
                collection=CACHE_COLLECTION_NAME,
            ),
            docstore_strategy=DOCSTORE_STRATEGY,
        )

        self.pipeline = pipeline

        return pipeline

    def run_pipeline(self) -> Sequence[BaseNode]:
        documents = self.reader.load_data()

        return self.pipeline.run(documents=documents, show_progress=True, num_workers=4)
