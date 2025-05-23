import logging
import os
import subprocess

from dotenv import load_dotenv
from llama_index.server import LlamaIndexServer, UIConfig
from prometheus_fastapi_instrumentator import Instrumentator
from traceloop.sdk import Traceloop

from app.settings import init_settings
from app.workflow import create_workflow
from backend.src.auth.constants import AuthType
from backend.src.auth.routes import router as auth_router
from configs.app_config import AUTH_TYPE

logger = logging.getLogger("uvicorn.debug")
load_dotenv()
init_settings()
# Setting up Traceloop for instrumentation
Traceloop.init()

# A path to a directory where the customized UI code is stored
COMPONENT_DIR = "frontend/components"


def create_app():
    env = os.environ.get("APP_ENV")

    app = LlamaIndexServer(
        workflow_factory=create_workflow,  # A factory function that creates a new workflow for each request
        ui_config=UIConfig(
            component_dir=COMPONENT_DIR,
            app_title="The Chatbot",
        ),
        env=env,
        title="RAG Server",
        logger=logger,
    )
    # You can also add custom routes to the app
    app.add_api_route("/api/health", lambda: {"message": "OK"}, status_code=200)

    if AUTH_TYPE == AuthType.DISABLED:
        pass

    if AUTH_TYPE == AuthType.BASIC:
        app.include_router(auth_router, prefix="/auth", tags=["auth"])
    return app


app = create_app()

# Initialize Prometheus and instrument the app
Instrumentator().instrument(app).expose(app)


def run(env: str):
    os.environ["APP_ENV"] = env
    app_host = os.getenv("APP_HOST", "0.0.0.0")
    app_port = os.getenv("APP_PORT", "8000")

    if env == "dev":
        subprocess.run(["fastapi", "dev", "--host", app_host, "--port", app_port])
    else:
        subprocess.run(["fastapi", "run", "--host", app_host, "--port", app_port])
