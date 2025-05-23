import json
import logging
import os

from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI

from app.settings import init_settings
from common.utils import load_yaml_file
from indexing.indexing_task_runner import IndexingTaskConfig, IndexingTaskRunner
from rag.index_manager import IndexManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def generate_index_to_pgvector():
    """
    Index the documents in the data directory.
    """
    from llama_index.core.readers import SimpleDirectoryReader

    from app.settings import init_settings

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


def generate_index_to_pgvector_via_indexing_task():
    """
    Index the documents in the data directory.
    """
    load_dotenv()
    init_settings()

    logger.info("Creating new index")
    conf = load_yaml_file("indexing/tasks/default.yaml")
    print(json.dumps(conf, indent=4))
    # load the documents and create the index
    IndexingTaskRunner(IndexingTaskConfig(**conf)).run()


def generate_ui_for_workflow():
    """
    Generate UI for UIEventData event in app/workflow.py
    """
    import asyncio

    from main import COMPONENT_DIR

    # To generate UI components for additional event types,
    # import the corresponding data model (e.g., MyCustomEventData)
    # and run the generate_ui_for_workflow function with the imported model.
    # Make sure the output filename of the generated UI component matches the event type (here `ui_event`)
    try:
        from app.workflow import UIEventData
    except ImportError:
        raise ImportError("Couldn't generate UI component for the current workflow.")
    from llama_index.server.gen_ui import generate_event_component

    # works also well with Claude 3.7 Sonnet or Gemini Pro 2.5
    llm = OpenAI(model="gpt-4.1")
    code = asyncio.run(generate_event_component(event_cls=UIEventData, llm=llm))
    with open(f"{COMPONENT_DIR}/ui_event.jsx", "w") as f:
        f.write(code)
