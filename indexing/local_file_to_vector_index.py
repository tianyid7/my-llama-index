import logging
import os

from dotenv import load_dotenv
from llama_index.core.readers import SimpleDirectoryReader

from app.settings import init_settings
from rag.index_manager import IndexManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def generate_index_to_pgvector():
    """
    Index the documents in the data directory.
    """
    load_dotenv()
    init_settings()

    logger.info("Creating new index")
    # load the documents and create the index
    reader = SimpleDirectoryReader(
        os.environ.get("DATA_DIR", "data"),
        recursive=True,
    )
    documents = reader.load_data()
    IndexManager().create_vector_index_with_docs(documents)
