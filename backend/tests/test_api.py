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
import verity.api.sessions as sessions
from verity.engine.index_types import ChunkIndex, IndexManifest
from verity.engine.models import Chunk, MediaType
from verity.llm.nvidia_provider import NvidiaLLMError


@pytest.fixture(autouse=True)
def pin_offline_providers(monkeypatch: pytest.MonkeyPatch):
    """This suite must never call a model provider.

    api.main loads .env at import (which is the product behaviour), so a real
    key may be in the environment here. These two variables pin the
    deterministic fakes for every test in this module; a future test that
    wants the real backends must say so explicitly.
    """
    monkeypatch.setenv("VERITY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("VERITY_EMBEDDING_PROVIDER", "fake")


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host="127.0.0.1",
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


def _insert_message(
    conn: psycopg2.extensions.connection,
    session_id: int,
    role: str,
    content: str,
    kind: str = "answer",
) -> int:
    """Write one message directly, bypassing the model entirely."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO messages (session_id, role, content, kind) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (session_id, role, content, kind),
        )
        message_id = cur.fetchone()[0]
    conn.commit()
    return message_id


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

    def test_list_sessions_is_empty_without_messages(
        self, client: TestClient, db_conn: psycopg2.extensions.connection
    ) -> None:
        """A session is created lazily on the first question.

        An empty session is not a conversation yet; listing it would put a
        blank, unopenable row in the chat list.
        """
        client.post("/api/sessions", json={"workspace_id": 1})

        response = client.get("/api/sessions?workspace_id=1")

        assert response.status_code == 200
        assert response.json()["sessions"] == []

    def test_list_sessions_titles_a_chat_with_its_first_question(
        self, client: TestClient, db_conn: psycopg2.extensions.connection
    ) -> None:
        session_id = client.post(
            "/api/sessions", json={"workspace_id": 1}
        ).json()["session_id"]
        _insert_message(db_conn, session_id, "user", "What is the maximum junction temperature?")
        _insert_message(db_conn, session_id, "assistant", "94 C.")

        response = client.get("/api/sessions?workspace_id=1")

        (item,) = response.json()["sessions"]
        assert item["session_id"] == session_id
        assert item["title"] == "What is the maximum junction temperature?"
        assert item["message_count"] == 2
        assert item["last_message_at"] is not None

    def test_a_long_first_question_is_cut_at_a_word(
        self, client: TestClient, db_conn: psycopg2.extensions.connection
    ) -> None:
        question = (
            "Does the datasheet describe the thermal shutdown behaviour when "
            "the board is operated continuously at the maximum rated junction "
            "temperature without any additional heatsink?"
        )
        session_id = client.post(
            "/api/sessions", json={"workspace_id": 1}
        ).json()["session_id"]
        _insert_message(db_conn, session_id, "user", question)

        (item,) = client.get("/api/sessions?workspace_id=1").json()["sessions"]

        assert item["title"].endswith("…")
        assert len(item["title"]) <= 81
        # Cut at a space, not mid-word: the last visible word is whole.
        assert question.startswith(item["title"][:-1].rstrip())
        assert not item["title"][:-1].endswith(" ")

    def test_list_sessions_is_newest_activity_first(
        self, client: TestClient, db_conn: psycopg2.extensions.connection
    ) -> None:
        older = client.post("/api/sessions", json={"workspace_id": 1}).json()["session_id"]
        newer = client.post("/api/sessions", json={"workspace_id": 1}).json()["session_id"]
        _insert_message(db_conn, older, "user", "first chat")
        _insert_message(db_conn, newer, "user", "second chat")
        # Activity, not creation: the older session is the one just used.
        _insert_message(db_conn, older, "user", "back to the first chat")

        sessions = client.get("/api/sessions?workspace_id=1").json()["sessions"]

        assert [item["session_id"] for item in sessions] == [older, newer]

    def test_list_sessions_is_scoped_to_its_workspace(
        self, client: TestClient, db_conn: psycopg2.extensions.connection
    ) -> None:
        mine = client.post("/api/sessions", json={"workspace_id": 1}).json()["session_id"]
        _insert_message(db_conn, mine, "user", "mine")
        with db_conn.cursor() as cur:
            cur.execute("INSERT INTO workspaces (id, name) VALUES (2, 'other')")
            cur.execute(
                "INSERT INTO sessions (workspace_id, state, summary) "
                "VALUES (2, '{}', NULL) RETURNING id"
            )
            other = cur.fetchone()[0]
        db_conn.commit()
        _insert_message(db_conn, other, "user", "theirs")

        sessions = client.get("/api/sessions?workspace_id=1").json()["sessions"]

        assert [item["session_id"] for item in sessions] == [mine]

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

    def test_model_failure_is_502_not_a_bare_500(
        self,
        client: TestClient,
        db_conn: psycopg2.extensions.connection,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An outage upstream is reported as such, not as "no evidence".

        A 500 hides which half failed. A refusal would be worse: it would
        claim the documents do not contain the answer when the model was never
        able to read them.
        """
        session_id = client.post(
            "/api/sessions", json={"workspace_id": 1}
        ).json()["session_id"]

        chunk = Chunk(
            rel_path="a.txt",
            media_type=MediaType.TEXT,
            location="text line 1",
            ordinal=1,
            text="the notice period is thirty days",
        )
        manifest = IndexManifest(
            format_version=1,
            created_utc="",
            corpus_fingerprint="",
            chunk_count=1,
            file_count=1,
            provider_id=None,
            dimension=None,
            status_counts={},
            skipped_count=0,
        )
        index = ChunkIndex(manifest=manifest, chunks=(chunk,), vectors=())
        monkeypatch.setattr(sessions, "load_all_chunks", lambda *a, **k: index)

        def fail_to_decide(self, *args, **kwargs):
            raise NvidiaLLMError("the provider returned an empty completion")

        monkeypatch.setattr(sessions.AgentLoop, "run", fail_to_decide)

        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"question": "What is the notice period?"},
        )

        assert response.status_code == 502
        assert "language model could not answer" in response.json()["detail"]


class TestIngestion:
    def test_ingest_nonexistent_path_returns_400(
        self, client: TestClient, db_conn: psycopg2.extensions.connection
    ) -> None:
        response = client.post(
            "/api/ingestion",
            json={"corpus_path": "/nonexistent/path"},
        )
        assert response.status_code == 400
