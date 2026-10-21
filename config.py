"""
config.py — Central configuration for the Research Assistant.

All settings live here so you only ever change one file.
Import this module wherever you need a setting.
"""

import os
from dotenv import load_dotenv

# Load variables from .env into os.environ
load_dotenv()

# ── LLM Settings ──────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = "gemini-2.5-flash"   # Fast + free-tier friendly

# ── Research Loop Settings ────────────────────────────────────────────
MAX_SEARCH_RESULTS: int = 5       # URLs fetched per sub-question
MAX_ITERATIONS: int = 3           # Max Critic→Researcher loops
MAX_SCRAPE_CHARS: int = 3000      # Characters kept per scraped page

# ── ChromaDB Settings ─────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = "./chroma_db"   # Where vectors are saved on disk
CHROMA_COLLECTION: str = "research_findings"

# ── SQLite Settings ───────────────────────────────────────────────────
SQLITE_DB_PATH: str = "./research_history.db"

# ── Embedding Model ───────────────────────────────────────────────────
# all-MiniLM-L6-v2: small (80 MB), fast on CPU, good quality
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ── Sanity Check ──────────────────────────────────────────────────────
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY is not set. "
        "Copy .env.example to .env and add your key."
    )