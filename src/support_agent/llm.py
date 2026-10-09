"""Create the chat model. The provider is a setting, so the same agent runs on Groq, Anthropic, OpenAI, ...

Environment variables (put them in a .env file, never in code):
    LLM_PROVIDER   default "groq"
    LLM_MODEL      default "openai/gpt-oss-120b"
    GROQ_API_KEY   (or the provider's own key variable, e.g. ANTHROPIC_API_KEY)
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL = "openai/gpt-oss-120b"


def get_llm(api_key: str | None = None):
    """api_key overrides the environment (used when a demo visitor pastes their own key)."""
    load_dotenv(ENV_FILE)
    provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    extra = {"api_key": api_key} if api_key else {}
    return init_chat_model(model, model_provider=provider, temperature=0, **extra)


def has_api_key() -> bool:
    load_dotenv(ENV_FILE)
    provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)
    return bool(os.environ.get(f"{provider.upper()}_API_KEY"))
