from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from prompts import Prompts


def create_react_agent(
        name: str,
        description: str,
        query_engine: BaseQueryEngine,
) -> ReActAgent:
    """
    Creates a ReAct agent from a given QueryEngine
    """
    query_engine_tools = [
        QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name=name,
                description=description,
            ),
        )
    ]

    llm = Settings.llm
    agent = ReActAgent.from_tools(
        query_engine_tools, llm=llm, verbose=True, context=Prompts.system_prompt
    )
    return agent
