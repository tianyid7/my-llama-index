import llama_index.core
import tiktoken
from llama_index.core import Settings
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
    TokenCountingHandler,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from app.prompts import Prompts


def init_settings(provider: str = "openai"):
    if provider == "openai":
        Settings.llm = OpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            system_prompt=Prompts.system_prompt,
            max_tokens=3000,
        )
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # Setting Callback Handlers
    llama_index.core.set_global_handler("simple")
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode,
        verbose=True,  # set to true to see usage printed to the console
    )

    Settings.callback_manager = CallbackManager([token_counter, LlamaDebugHandler()])
