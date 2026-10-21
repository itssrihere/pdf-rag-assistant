import pytest

from config import _build_db_url


def test_db_url_wins_when_set():
    env = {"DB_URL": "postgresql+psycopg://u:p@h:1/d", "POSTGRES_PASSWORD": "ignored"}
    assert _build_db_url(env) == "postgresql+psycopg://u:p@h:1/d"


def test_builds_url_with_defaults():
    assert _build_db_url({"POSTGRES_PASSWORD": "pw"}) == "postgresql+psycopg://postgres:pw@localhost:5432/rag"


def test_builds_url_from_all_parts():
    env = {
        "POSTGRES_USER": "me",
        "POSTGRES_PASSWORD": "pw",
        "POSTGRES_HOST": "db",
        "POSTGRES_PORT": "6543",
        "POSTGRES_DB": "papers",
    }
    assert _build_db_url(env) == "postgresql+psycopg://me:pw@db:6543/papers"


def test_special_characters_in_credentials_are_encoded():
    url = _build_db_url({"POSTGRES_USER": "a@b", "POSTGRES_PASSWORD": "p@ss/w:rd"})
    assert url == "postgresql+psycopg://a%40b:p%40ss%2Fw%3Ard@localhost:5432/rag"


def test_missing_password_raises_clear_error():
    with pytest.raises(RuntimeError, match="POSTGRES_PASSWORD"):
        _build_db_url({})


def test_empty_db_url_falls_back_to_parts():
    assert _build_db_url({"DB_URL": "", "POSTGRES_PASSWORD": "pw"}).startswith("postgresql+psycopg://postgres:pw@")
