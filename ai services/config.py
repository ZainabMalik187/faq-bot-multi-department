"""
config.py — Centralised credentials and configuration.

All sensitive values are loaded from environment variables.
Update this file to add new API keys or model settings.
"""

import os
from dotenv import load_dotenv

# Load .env file if present (for local development)
load_dotenv(override=True)

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# API Keys (with rotation support)
# ---------------------------------------------------------------------------
GROQ_API_KEY1: str = os.environ.get("GROQ_API_KEY1", "")
GROQ_API_KEY2: str = os.environ.get("GROQ_API_KEY2", "")
GROQ_API_KEY3: str = os.environ.get("GROQ_API_KEY3", "")

GROQ_API_KEYS: list[str] = [k for k in [GROQ_API_KEY1, GROQ_API_KEY2, GROQ_API_KEY3] if k]

if not GROQ_API_KEYS:
    default_key = os.environ.get("GROQ_API_KEY", "")
    if default_key:
        GROQ_API_KEYS = [default_key]

_current_key_idx = 0

def get_current_key() -> str:
    if not GROQ_API_KEYS:
        return ""
    return GROQ_API_KEYS[_current_key_idx % len(GROQ_API_KEYS)]

GROQ_API_KEY: str = get_current_key()

def rotate_key() -> str:
    global _current_key_idx, GROQ_API_KEY
    if not GROQ_API_KEYS:
        return ""
    _current_key_idx += 1
    new_key = get_current_key()
    GROQ_API_KEY = new_key
    print(f"[INFO] Rotating Groq API Key to index {_current_key_idx % len(GROQ_API_KEYS)}", flush=True)
    return new_key

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------
GROQ_MODEL: str = "llama-3.1-8b-instant"
EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
SIMILARITY_THRESHOLD: float = 0.40  # Set to 0.40 to allow highly rephrased valid queries

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# PostgreSQL Configuration (pgvector)
# ---------------------------------------------------------------------------
PG_HOST: str = os.environ.get("PG_HOST", "localhost")
PG_PORT: int = int(os.environ.get("PG_PORT", 5432))
PG_DB: str = os.environ.get("PG_DB", "faq_db")
PG_USER: str = os.environ.get("PG_USER", "postgres")
PG_PASSWORD: str = os.environ.get("PG_PASSWORD", "postgres")

FAQ_FILE: str = os.path.join("data", "faq.json")
