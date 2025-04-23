from typing import Optional

from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
from llama_index.server.api.models import ChatRequest
from llama_index.server.tools.index import get_query_engine_tool

from app.index import get_index
from rag.index_manager import IndexManager
from rag.query_engine_manager import QueryEngineManager


def create_workflow(chat_request: Optional[ChatRequest] = None) -> AgentWorkflow:
    index = get_index(chat_request=chat_request)
    if index is None:
        raise RuntimeError(
            "Index not found! Please run `poetry run generate` to index the data first."
        )
    query_tool = get_query_engine_tool(index=index)
    return AgentWorkflow.from_tools_or_functions(
        tools_or_functions=[query_tool],
        llm=Settings.llm or OpenAI(model="gpt-4o-mini"),
        system_prompt="You are a helpful assistant.",
    )


index = IndexManager(
    pgvector_conn="postgresql://postgres:postgres@localhost:5431/vectordb",
    pgvector_table="document",
    redis_host="localhost",
).create_vector_index()

query_tool = QueryEngineManager(base_index=index).get_query_engine_tool(
    name="query_tool",
    description="Use this tool to retrieve information about the text corpus from an index.",
)


def create_workflow_v1(chat_request: Optional[ChatRequest] = None) -> AgentWorkflow:
    if index is None:
        raise RuntimeError(
            "Index not found! Please run `poetry run generate` to index the data first."
        )

    return AgentWorkflow.from_tools_or_functions(
        tools_or_functions=[query_tool],
        llm=Settings.llm or OpenAI(model="gpt-4o-mini"),
        system_prompt="You are a helpful assistant.",
    )
