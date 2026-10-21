import os

# config.py reads these at import time; set them before any test module imports it.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("POSTGRES_PASSWORD", "test-password")
