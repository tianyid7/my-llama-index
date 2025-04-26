import logging

from dotenv import load_dotenv

from app.settings import init_settings
from common.utils import load_yaml_file
from indexing.indexing_task_runner import IndexingTaskConfig, IndexingTaskRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


if __name__ == "__main__":
    """
    Index the documents in the data directory.
    """
    load_dotenv()
    init_settings()

    logger.info("Creating new index")
    conf = load_yaml_file("default.yaml")
    IndexingTaskRunner(IndexingTaskConfig(**conf)).run()
