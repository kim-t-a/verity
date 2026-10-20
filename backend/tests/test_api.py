"""Tests for the Verity API.

Uses FastAPI's TestClient — no running server needed. Tests that write to
PostgreSQL are skipped when the database is not available.

Tests cover:
- Health check
- Document upload, list, detail, delete
- Session creation
- Ask question → answer / clarification / refusal
- Ingestion trigger
"""

from __future__ import annotations

import psycopg2
import pytest
from fastapi.testclient import TestClient

from verity.api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="verity",
        user="verity",
        password="verity_password",
    )


@pytest.fixture
def db_conn() -> psycopg2.extensions.connection:
    try:
        conn = _connect()
    except psycopg2.OperationalError:
        pytest.skip("PostgreSQL not available")
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def clean_api_data(db_conn: psycopg2.extensions.connection):
    """Clean all data before each API test."""
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM citations")
        cur.execute("DELETE FROM messages")
        cur.execute("DELETE FROM sessions")
        cur.execute("DELETE FROM chunks")
        cur.execute("DELETE FROM document_versions")
        cur.execute("DELETE FROM documents")
        cur.execute("DELETE FROM workspaces")
        # Re-insert the default workspace.
        cur.execute("INSERT INTO workspaces (id, name) VALUES (1, 'default')")
    db_conn.commit()
    yield


class TestHealth:
    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestDocuments:
    def test_upload_document_returns_202(
        self, client: TestClient, db_conn: psycopg2.extensions.connection
    ) -> None:
        response = client.post(
            "/api/workspaces/1/documents",
            files={"file": ("test.txt", b"Hello world\n", "text/plain")},
        )
        assert response.status_code == 202
        data = response.json()
        assert data["duplicate"] is False
        assert "document_id" in data
        assert data["status"] == "pending"

    def test_upload_duplicate_document(
        self, client: TestClient, db_conn: psycopg2.extensions.connection
    ) -> None:
        # Upload the same content twice.
        response1 = client.post(
            "/api/workspaces/1/documents",
            files={"file": ("test.txt", b"duplicate content\n", "text/plain")},
        )
        assert response1.status_code == 202
        doc_id = response1.json()["document_id"]

        response2 = client.post(
            "/api/workspaces/1/documents",
            files={"file": ("test2.txt", b"duplicate content\n", "text/plain")},
        )
        assert response2.status_code == 202
        assert response2.json()["duplicate"] is True
        assert response2.json()["document_id"] == doc_id

    def test_list_documents(self, client: TestClient, db_conn: psycopg2.extensions.connection) -> None:
        client.post(
            "/api/workspaces/1/documents",
            files={"file": ("a.txt", b"content a\n", "text/plain")},
        )
        client.post(
            "/api/workspaces/1/documents",
            files={"file": ("b.txt", b"content b\n", "text/plain")},
        )

        response = client.get("/api/workspaces/1/documents")
        assert response.status_code == 200
        data = response.json()
        assert len(data["documents"]) == 2

    def test_get_document_detail(
        self, client: TestClient, db_conn: psycopg2.extensions.connection
    ) -> None:
        upload_resp = client.post(
            "/api/workspaces/1/documents",
            files={"file": ("detail.txt", b"detail content\n", "text/plain")},
        )
        doc_id = upload_resp.json()["document_id"]

        response = client.get(f"/api/workspaces/1/documents/{doc_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == doc_id
        assert data["filename"] == "detail.txt"
        assert data["status"] in ("pending", "processing", "ready", "failed")

    def test_delete_document(
        self, client: TestClient, db_conn: psycopg2.extensions.connection
    ) -> None:
        upload_resp = client.post(
            "/api/workspaces/1/documents",
            files={"file": ("todelete.txt", b"delete me\n", "text/plain")},
        )
        doc_id = upload_resp.json()["document_id"]

        response = client.delete(f"/api/workspaces/1/documents/{doc_id}")
        assert response.status_code == 204

        # Verify it's gone.
        response = client.get(f"/api/workspaces/1/documents/{doc_id}")
        assert response.status_code == 404

    def test_delete_nonexistent_document_returns_404(
        self, client: TestClient, db_conn: psycopg2.extensions.connection
    ) -> None:
        response = client.delete("/api/workspaces/1/documents/99999")
        assert response.status_code == 404


class TestSessions:
    def test_create_session(self, client: TestClient, db_conn: psycopg2.extensions.connection) -> None:
        response = client.post(
            "/api/sessions",
            json={"workspace_id": 1},
        )
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert isinstance(data["session_id"], int)

    def test_ask_question_no_documents(
        self, client: TestClient, db_conn: psycopg2.extensions.connection
    ) -> None:
        # Create a session first.
        session_resp = client.post("/api/sessions", json={"workspace_id": 1})
        session_id = session_resp.json()["session_id"]

        # Ask without any ingested documents.
        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"question": "What is the value?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "refusal"
        assert "No documents" in data["text"]

    def test_ask_nonexistent_session_returns_404(
        self, client: TestClient, db_conn: psycopg2.extensions.connection
    ) -> None:
        response = client.post(
            "/api/sessions/99999/messages",
            json={"question": "Hello?"},
        )
        assert response.status_code == 404

    def test_get_messages_empty(
        self, client: TestClient, db_conn: psycopg2.extensions.connection
    ) -> None:
        session_resp = client.post("/api/sessions", json={"workspace_id": 1})
        session_id = session_resp.json()["session_id"]

        response = client.get(f"/api/sessions/{session_id}/messages")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert data["messages"] == []


class TestIngestion:
    def test_ingest_nonexistent_path_returns_400(
        self, client: TestClient, db_conn: psycopg2.extensions.connection
    ) -> None:
        response = client.post(
            "/api/ingestion",
            json={"corpus_path": "/nonexistent/path"},
        )
        assert response.status_code == 400
