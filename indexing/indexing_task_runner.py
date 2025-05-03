import logging
from typing import Optional

from llama_index.core import PropertyGraphIndex, Settings
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from pydantic import BaseModel

from common.decorators import log_function_time
from configs.rag_config import (
    CACHE_COLLECTION_NAME,
    DOCSTORE_STRATEGY,
    REDIS_HOST,
    REDIS_PORT,
)
from indexing.readers.helper import prepare_reader
from indexing.transformations.helper import (
    TransformationConfig,
    prepare_transformations,
)
from rag.index_manager import IndexManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndexingTaskConfig(BaseModel):
    """
    Configuration for an indexing task.
    """

    reader: str
    """
    Reader to use for loading documents.
    """

    reader_config: dict
    """
    Parameters for the reader.
    """

    transformations: list[str]
    """
    Transformations to apply to the ingestion pipeline. Order matters.
    See PROVIDED_TRANSFORMATIONS for a list of transformations.
    """

    transformation_config: TransformationConfig
    """
    Parameters for the transformations.
    """

    num_workers: int = 4
    """
    Number of workers to use for the ingestion task.
    """

    runner: str = "default"
    """
    Runner to use for the indexing task. Value can be default or ray.
    default: use the default runner by llama_index ingestion pipeline
    ray: use the ray data to achieve the parallelism
    """

    graph_index: bool = False
    """
    Whether to create a graph index or not. Default is False.
    If True, the graph index will be created after the base vector index is created.
    """


class IndexingTaskRunner:
    """
    A manager for the indexing task, with the reader, transformations.
    """

    def __init__(self, config: IndexingTaskConfig):
        self.config = config

        self.reader = prepare_reader(
            self.config.reader, reader_config=self.config.reader_config
        )
        logger.info(
            f"Prepared the reader successfully: {self.config.reader}({self.reader.__class__.__name__})"
        )

        self.pipeline: Optional[IngestionPipeline] = None

        self.base_index = IndexManager().base_index
        self.property_graph_index = IndexManager().property_graph_index

        if self.config.runner == "ray":
            raise NotImplementedError(
                "Ray runner is not implemented yet. Please use the default runner."
            )
        elif self.config.runner == "default":
            logger.info("Using the default runner for the indexing task...")
            self._build_default_runner()
        else:
            raise ValueError(
                f"Invalid runner: {self.config.runner}. Supported runners are default and ray."
            )

    def _build_default_runner(self):
        """
        Build a default runner. This is the default runner by llama_index ingestion pipeline.
        """

        """
        Prepare transformations for the ingestion pipeline.
        """
        transformations = prepare_transformations(
            transformations=self.config.transformations,
            config=self.config.transformation_config,
        )
        logger.info(
            f"Prepared transformations successfully: {self.config.transformations}"
        )

        """
        Build the ingestion pipeline.
        """
        pipeline = IngestionPipeline(
            transformations=transformations,
            vector_store=self.base_index.vector_store,
            docstore=self.base_index.docstore,
            # use cache only if parallelism is not activated (num_worker < 2)
            # due to the issue "TypeError: cannot pickle '_thread.lock' object"
            cache=IngestionCache(
                cache=RedisCache.from_host_and_port(REDIS_HOST, REDIS_PORT),
                collection=CACHE_COLLECTION_NAME,
            )
            if self.config.num_workers == 1
            else None,
            docstore_strategy=DOCSTORE_STRATEGY,
        )
        self.pipeline = pipeline
        logger.info(f"Built the ingestion pipeline successfully!")

    @log_function_time
    def run(self, disable_cache: bool = False):
        if disable_cache:
            logger.info("Disabling cache for the indexing task...")
            self.pipeline.disable_cache = True

        logger.info("Starting indexing task...")
        documents = self.reader.load_data()
        logger.info(f"Loaded {len(documents)} documents from the reader.")
        result = self.pipeline.run(
            documents=documents, show_progress=True, num_workers=self.config.num_workers
        )

        if not result:
            logger.warning("No nodes were created from the documents.")
            return

        if "hierarchical" in self.config.transformations:
            logger.info(
                "HierarchicalNodeParser is applied. Storing nodes in docstore for auto_merging retriever..."
            )
            self.base_index.docstore.add_documents(result)

        logger.info(
            f"Indexing task completed successfully. {len(result)} nodes indexed."
        )

        # If the graph index is enabled, the graph index will be created after the base vector index is created.
        if self.config.graph_index:
            self.property_graph_index.from_documents(
                documents,
                property_graph_store=self.property_graph_index.property_graph_store,
                show_progress=True,
            )
