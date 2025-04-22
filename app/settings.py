from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from prompts import Prompts


def init_settings(provider: str = "openai"):
    if provider == "openai":
        Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.0, system_prompt=Prompts.system_prompt, max_tokens=3000)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
