import os
from collections.abc import Mapping
from urllib.parse import quote_plus

from dotenv import load_dotenv

load_dotenv()  # reads .env if present; real environment variables take precedence


def _build_db_url(env: Mapping[str, str] = os.environ) -> str:
    """DB_URL wins if set; otherwise assemble it from the POSTGRES_* variables."""
    if env.get("DB_URL"):
        return env["DB_URL"]
    password = env.get("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError("Set DB_URL, or POSTGRES_PASSWORD (e.g. in .env) to connect to Postgres")
    user = env.get("POSTGRES_USER", "postgres")
    host = env.get("POSTGRES_HOST", "localhost")
    port = env.get("POSTGRES_PORT", "5432")
    db = env.get("POSTGRES_DB", "rag")
    return f"postgresql+psycopg://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{db}"


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]  # embeddings (and chat if CHAT_PROVIDER=openai)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")  # chat, when CHAT_PROVIDER=anthropic
CHAT_PROVIDER = os.environ.get("CHAT_PROVIDER", "anthropic")  # anthropic | openai
CHAT_MODEL = os.environ.get("CHAT_MODEL")  # default depends on the provider (see llm.py)
DB_URL = _build_db_url()
DOCS_DIR = os.environ.get("DOCS_DIR")  # no default — must be passed via --docs-dir or env var
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))
RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K", "8"))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "pdf_docs")
PDF_EXTRACTOR = os.environ.get("PDF_EXTRACTOR", "pypdf")  # pypdf | pypdfium2 (see pdf_text.py)
HISTORY_TURNS = int(os.environ.get("HISTORY_TURNS", "5"))  # past Q&A pairs remembered in chat

