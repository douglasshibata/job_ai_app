import os
import sys
from pathlib import Path

# Ensure directory and parent directory are on sys.path for seamless imports
file_dir = Path(__file__).resolve().parent
if str(file_dir) not in sys.path:
    sys.path.insert(0, str(file_dir))
if str(file_dir.parent) not in sys.path:
    sys.path.insert(0, str(file_dir.parent))

from dotenv import load_dotenv

# Load .env file if available
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# LLM Provider API Keys and Configurations
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

PRIMARY_LLM_PROVIDER = os.getenv("PRIMARY_LLM_PROVIDER", "groq").lower()

# Database Configuration
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "job_ai_app.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

# Search Defaults
DEFAULT_SEARCH_TERM = os.getenv("DEFAULT_SEARCH_TERM", "Software Engineer")
DEFAULT_LOCATION = os.getenv("DEFAULT_LOCATION", "Remote")
DEFAULT_HOURS_OLD = int(os.getenv("DEFAULT_HOURS_OLD", "72"))
DEFAULT_RESULTS_WANTED = int(os.getenv("DEFAULT_RESULTS_WANTED", "15"))
DEFAULT_SITES = ["linkedin", "indeed", "glassdoor", "zip_recruiter"]
