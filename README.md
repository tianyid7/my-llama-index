This is a [LlamaIndex](https://www.llamaindex.ai/) simple agentic RAG project using [Agent Workflows](https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_basic/).

## Getting Started

First, setup the environment with poetry:

> **_Note:_** This step is not needed if you are using the dev-container.

```shell
poetry install
```

Then check the parameters that have been pre-configured in the `.env` file in this directory.
Make sure you have set the `OPENAI_API_KEY` for the LLM.

If want to use LangSmith for observability, add below Traceloop env vars to allow tracings to be sent to LangSmith:
```shell
TRACELOOP_BASE_URL=https://api.smith.langchain.com/otel
TRACELOOP_HEADERS="x-api-key=<LANGSMITH_API_KEY>,Langsmith-Project=<LANGSMITH_PROJECT_NAME>"
````
Refer to [LangSmith docs](https://docs.smith.langchain.com/observability/how_to_guides/trace_with_opentelemetry#logging-traces-with-the-traceloop-sdk) for more details.

Second, start databases in docker compose.
```shell
docker compose up -d
```

Third, generate the embeddings of the documents in the `./data` directory. This will store embeddings in the vector database:

```shell
poetry run generate
```

Lastly, run the development server:

```shell
poetry run dev
```

Then open [http://localhost:8000](http://localhost:8000) with your browser to start the chat UI.

To start the app optimized for **production**, run:

```
poetry run prod
```

## Configure LLM and Embedding Model

You can configure [LLM model](https://docs.llamaindex.ai/en/stable/module_guides/models/llms) and [embedding model](https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings) in [settings.py](app/settings.py).

## Use Case

We have prepared an [example workflow](./app/workflow.py) for the agentic RAG use case, where you can ask questions about the example documents in the [./data](./data) directory.

You can start by sending an request on the [chat UI](http://localhost:8000) or you can test the `/api/chat` endpoint with the following curl request:

```
curl --location 'localhost:8000/api/chat' \
--header 'Content-Type: application/json' \
--data '{ "messages": [{ "role": "user", "content": "What standards for a letter exist?" }] }'
```

## Learn More

To learn more about LlamaIndex, take a look at the following resources:

- [LlamaIndex Documentation](https://docs.llamaindex.ai) - learn about LlamaIndex.
- [Workflows Introduction](https://docs.llamaindex.ai/en/stable/understanding/workflows/) - learn about LlamaIndex workflows.

You can check out [the LlamaIndex GitHub repository](https://github.com/run-llama/llama_index) - your feedback and contributions are welcome!
