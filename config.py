"""Application configuration loaded from environment variables via Pydantic Settings."""

from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()


class Config(BaseSettings):
    """Centralised configuration for the dual AI assistant system.

    All values are read from environment variables (or .env file).
    Defaults are provided for optional fields.
    """

    GROQ_API_KEY: str = Field(..., description="Groq API key for frontier model access")
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key for content moderation")
    HF_TOKEN: str = Field(default="", description="HuggingFace token (optional fallback)")
    MODAL_ENDPOINT: str = Field(
        default="https://your-modal-endpoint.modal.run",
        description="Deployed Modal HTTPS endpoint URL",
    )
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Base URL for local Ollama server",
    )
    OSS_MODEL_NAME: str = Field(
        default="qwen2.5:0.5b",
        description="Model name used by the OSS assistant (Ollama tag or HF id)",
    )
    FRONTIER_MODEL_NAME: str = Field(
        default="llama-3.3-70b-versatile",
        description="Model name for the Groq frontier assistant",
    )
    LLAMAGUARD_MODEL: str = Field(
        default="llama-3.1-8b-instant",
        description="Model used for LlamaGuard-style content moderation via Groq",
    )
    CONVERSATION_MAX_TURNS: int = Field(
        default=10,
        description="Maximum conversation turns kept in sliding-window memory",
    )
    TOXICITY_THRESHOLD: float = Field(
        default=0.7,
        description="Detoxify score above which content is considered toxic",
    )
    USE_MODAL: bool = Field(
        default=False,
        description="If True, route OSS calls to Modal endpoint; otherwise use Ollama",
    )
    LOG_LEVEL: str = Field(default="INFO", description="Python logging level string")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


config = Config()
