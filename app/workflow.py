from typing import Optional

from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
from llama_index.server.api.models import ChatRequest

from app.settings import init_settings
from rag.index_manager import IndexManager
from rag.query_engine_manager import QueryEngineManager

init_settings()  # Initialize RAG settings. Must be placed at the beginning of RAG program lifecycle.
index = IndexManager().base_index

query_tool = QueryEngineManager(base_index=index).get_query_engine_tool(
    name="query_tool",
    description="Use this tool to retrieve information about the text corpus from an index.",
)


def create_workflow(chat_request: Optional[ChatRequest] = None) -> AgentWorkflow:
    if index is None:
        raise RuntimeError(
            "Index not found! Please run `poetry run generate` to index the data first."
        )

    return AgentWorkflow.from_tools_or_functions(
        tools_or_functions=[query_tool],
        llm=Settings.llm or OpenAI(model="gpt-4o-mini"),
        system_prompt="You are a helpful assistant.",
    )
