"""Alembic environment for Verity database migrations.

Uses SQLAlchemy with psycopg2 to connect to PostgreSQL. Migrations are
written as raw SQL via ``op.execute()`` since we don't have SQLAlchemy
model metadata yet.
"""

from __future__ import annotations

import os
from logging.config import fileConfig
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, pool

from alembic import context

# Load the repository-root `.env` for local runs, without overriding values
# that are already set in the process environment.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# No SQLAlchemy model metadata -- migrations are raw SQL.
target_metadata = None


def _database_url() -> str:
    """The database URL Alembic connects with.

    Docker Compose sets ``VERITY_DB_*`` for the backend container, where the
    database is reachable as ``db`` rather than ``127.0.0.1``. The value in
    ``alembic.ini`` is the local fallback. Empty values (as shipped in
    ``.env.example``) count as unset, so they cannot blank out the defaults.
    """
    host = os.environ.get("VERITY_DB_HOST", "").strip()
    if not host:
        return config.get_main_option("sqlalchemy.url")

    port = os.environ.get("VERITY_DB_PORT", "").strip() or "5432"
    dbname = os.environ.get("VERITY_DB_NAME", "").strip() or "verity"
    user = os.environ.get("VERITY_DB_USER", "").strip() or "verity"
    password = os.environ.get("VERITY_DB_PASSWORD", "").strip() or "verity_password"
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Emits SQL to stdout without connecting to the database.
    """
    url = _database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode using SQLAlchemy engine."""
    connectable = create_engine(_database_url(), poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
