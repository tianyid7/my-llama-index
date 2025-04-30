from dotenv import load_dotenv

from rag.query_engine_manager import QueryEngineManager


async def run_query():
    """
    Run a query against the index.
    """
    from app.settings import init_settings
    from rag.index_manager import IndexManager

    load_dotenv()
    init_settings()

    index = IndexManager().base_index

    query_engine = QueryEngineManager(base_index=index).query_engine
    response = await query_engine.aquery("ZhangSan weight")
    print(response)


def main():
    import asyncio

    asyncio.run(run_query())
