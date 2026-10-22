"""Database connection and session management for Verity.

Provides a PostgreSQL connection pool and session helpers used by the
ingestion worker, the API layer, and memory service. The engine
(``verity.engine``) remains pure standard library and knows nothing about
this module -- it is the product layer that owns the database.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg2
from psycopg2 import pool

#: Connection parameters read from environment with sensible local defaults.
#: The default host is 127.0.0.1, not "localhost": on Windows, psycopg2 tries
#: "localhost" as IPv6 ::1 first and each publication that doesn't answer
#: costs the full TCP fallback timeout (~21 s per connection, measured
#: 2026-10-21). Docker publishes the port on all interfaces, so the loopback
#: address is both correct and ~1000x faster.
#: The API key and other secrets live in .env and are never hardcoded here.
def _env(name: str, default: str) -> str:
    """Read an environment value, treating an empty string as unset.

    ``.env.example`` ships every variable with no value, and python-dotenv
    loads those as empty strings. ``os.environ.get(name, default)`` would then
    return ``""`` instead of the default -- and ``int("")`` for the port raises
    before the application can start.
    """
    value = os.environ.get(name, "").strip()
    return value or default


DB_CONFIG = {
    "host": _env("VERITY_DB_HOST", "127.0.0.1"),
    "port": int(_env("VERITY_DB_PORT", "5432")),
    "dbname": _env("VERITY_DB_NAME", "verity"),
    "user": _env("VERITY_DB_USER", "verity"),
    "password": _env("VERITY_DB_PASSWORD", "verity_password"),
}

#: Minimum and maximum connections in the pool.
MIN_CONN = 1
MAX_CONN = 10

_connection_pool: pool.AbstractConnectionPool | None = None


def get_connection_pool() -> pool.AbstractConnectionPool:
    """Return the shared connection pool, creating it on first use."""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = pool.ThreadedConnectionPool(
            MIN_CONN, MAX_CONN, **DB_CONFIG
        )
    return _connection_pool


@contextmanager
def get_connection() -> Iterator[psycopg2.extensions.connection]:
    """Yield a pooled connection, returning it to the pool on exit.

    The connection is rolled back on exception and committed on success,
    so callers don't need to manage transactions explicitly for simple
    read/write operations.
    """
    pool = get_connection_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def close_pool() -> None:
    """Close all pooled connections. Mostly for tests and shutdown."""
    global _connection_pool
    if _connection_pool is not None:
        _connection_pool.closeall()
        _connection_pool = None


def ensure_vector_extension() -> None:
    """Ensure the pgvector extension is installed in the database."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
