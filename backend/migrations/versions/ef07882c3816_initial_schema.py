"""initial_schema

Revision ID: ef07882c3816
Revises:
Create Date: 2026-10-20 15:28:02.067374

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'ef07882c3816'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIMENSION = 2048


def upgrade() -> None:
    """Create the initial schema."""

    # workspaces: one shared workspace for now (auth deferred).
    op.execute("""
        CREATE TABLE workspaces (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL DEFAULT 'default',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("INSERT INTO workspaces (name) VALUES ('default')")

    # documents: one row per uploaded document.
    op.execute("""
        CREATE TABLE documents (
            id SERIAL PRIMARY KEY,
            workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            filename TEXT NOT NULL,
            storage_path TEXT NOT NULL,
            content_type TEXT NOT NULL,
            size_bytes BIGINT NOT NULL,
            sha256 TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            processed_at TIMESTAMPTZ,
            CONSTRAINT documents_sha256_workspace_unique UNIQUE (sha256, workspace_id)
        )
    """)
    op.execute("CREATE INDEX idx_documents_workspace ON documents(workspace_id)")
    op.execute("CREATE INDEX idx_documents_status ON documents(status)")
    op.execute("CREATE INDEX idx_documents_sha256 ON documents(sha256)")

    # document_versions: tracks changes so derived memories/index can be invalidated.
    op.execute("""
        CREATE TABLE document_versions (
            id SERIAL PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            version INTEGER NOT NULL DEFAULT 1,
            sha256 TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX idx_document_versions_document ON document_versions(document_id)")

    # chunks: the hybrid index — tsv for BM25-equivalent FTS, embedding for dense search.
    op.execute(f"""
        CREATE TABLE chunks (
            id SERIAL PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            version_id INTEGER REFERENCES document_versions(id) ON DELETE SET NULL,
            ordinal INTEGER NOT NULL,
            text TEXT NOT NULL,
            location TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            embedding vector({EMBEDDING_DIMENSION}),
            tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
        )
    """)
    op.execute("CREATE INDEX idx_chunks_document ON chunks(document_id)")
    op.execute("CREATE INDEX idx_chunks_version ON chunks(version_id)")
    op.execute("CREATE INDEX idx_chunks_tsv ON chunks USING gin (tsv)")
    op.execute("CREATE INDEX idx_chunks_ordinal ON chunks(document_id, ordinal)")

    # sessions: conversation state per workspace.
    op.execute("""
        CREATE TABLE sessions (
            id SERIAL PRIMARY KEY,
            workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            state JSONB NOT NULL DEFAULT '{}',
            summary TEXT
        )
    """)
    op.execute("CREATE INDEX idx_sessions_workspace ON sessions(workspace_id)")

    # messages: conversation history with discriminated type.
    op.execute("""
        CREATE TABLE messages (
            id SERIAL PRIMARY KEY,
            session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            kind TEXT NOT NULL CHECK (kind IN ('answer', 'clarification', 'refusal')),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX idx_messages_session ON messages(session_id)")
    op.execute("CREATE INDEX idx_messages_created ON messages(created_at)")

    # citations: audit trail linking messages to chunks/documents.
    op.execute("""
        CREATE TABLE citations (
            message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            PRIMARY KEY (message_id, chunk_id)
        )
    """)
    op.execute("CREATE INDEX idx_citations_document ON citations(document_id)")

    # memories: user-controlled memory records (longterm, document, conversation).
    op.execute("""
        CREATE TABLE memories (
            id SERIAL PRIMARY KEY,
            workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            type TEXT NOT NULL CHECK (type IN ('longterm', 'document', 'conversation')),
            content TEXT NOT NULL,
            source_provenance TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            reviewed_at TIMESTAMPTZ,
            retention_expires_at TIMESTAMPTZ,
            user_controlled BOOLEAN NOT NULL DEFAULT TRUE,
            source_document_id INTEGER REFERENCES documents(id) ON DELETE SET NULL
        )
    """)
    op.execute("CREATE INDEX idx_memories_workspace ON memories(workspace_id)")
    op.execute("CREATE INDEX idx_memories_type ON memories(type)")

    # jobs: ingestion queue using FOR UPDATE SKIP LOCKED (no Redis needed).
    op.execute("""
        CREATE TABLE jobs (
            id SERIAL PRIMARY KEY,
            kind TEXT NOT NULL,
            payload JSONB NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            run_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            locked_at TIMESTAMPTZ,
            error TEXT
        )
    """)
    op.execute("CREATE INDEX idx_jobs_status_run_at ON jobs(status, run_at)")


def downgrade() -> None:
    """Drop all tables in reverse order of creation."""
    op.execute("DROP TABLE IF EXISTS jobs")
    op.execute("DROP TABLE IF EXISTS memories")
    op.execute("DROP TABLE IF EXISTS citations")
    op.execute("DROP TABLE IF EXISTS messages")
    op.execute("DROP TABLE IF EXISTS sessions")
    op.execute("DROP TABLE IF EXISTS chunks")
    op.execute("DROP TABLE IF EXISTS document_versions")
    op.execute("DROP TABLE IF EXISTS documents")
    op.execute("DROP TABLE IF EXISTS workspaces")
