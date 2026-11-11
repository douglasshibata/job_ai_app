"""Configuration settings management via Pydantic BaseSettings."""
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

class Settings(BaseSettings):
    PRIMARY_LLM_PROVIDER: str = Field(default="groq")
    GROQ_API_KEY: str = Field(default="")
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile")
    GEMINI_API_KEY: str = Field(default="")
    GEMINI_MODEL: str = Field(default="gemini-2.5-flash")
    OLLAMA_HOST: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL: str = Field(default="llama3")

    DATABASE_URL: str = Field(default=f"sqlite:///{BASE_DIR / 'job_ai_platform.db'}")

    SCRAPER_WORKERS: int = Field(default=4)
    DEFAULT_SEARCH_TERM: str = Field(default="Software Engineer")
    DEFAULT_LOCATION: str = Field(default="Remote")
    DEFAULT_HOURS_OLD: int = Field(default=72)
    DEFAULT_RESULTS_WANTED: int = Field(default=15)
    DEFAULT_SITES: List[str] = Field(default_factory=lambda: ["linkedin", "indeed", "glassdoor", "zip_recruiter"])

    EMBEDDING_MODEL_NAME: str = Field(default="BAAI/bge-small-en-v1.5")
    VECTOR_DIMENSION: int = Field(default=384)
    TOP_PERCENTILE_LLM_FILTER: float = Field(default=0.15)

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH) if ENV_PATH.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
