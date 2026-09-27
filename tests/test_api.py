"""End-to-end tests for the MSN AI HTTP API."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.errors import MSNError, SessionNotFoundError, register_error_handlers
from app.main import create_app
from app.core.session import Session, SessionManager


def test_health_endpoint() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["app"] == "MSN AI"
        assert body["version"] == "0.1.0"
        assert body["provider"] == "mock"


def test_root_endpoint() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["service"] == "MSN AI"


def test_chat_creates_new_session() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.post("/api/v1/chat", json={"message": "Hello MSN AI"})
        assert response.status_code == 200
        body = response.json()
        assert body["session_id"]
        assert body["provider"] == "mock"
        assert "Hello MSN AI" in body["reply"]


def test_chat_reuses_session_and_keeps_history() -> None:
    app = create_app()
    with TestClient(app) as client:
        first = client.post("/api/v1/chat", json={"message": "turn one"})
        session_id = first.json()["session_id"]

        second = client.post(
            "/api/v1/chat", json={"message": "turn two", "session_id": session_id}
        )
        assert second.status_code == 200
        assert second.json()["session_id"] == session_id

        history = client.get(f"/api/v1/sessions/{session_id}")
        messages = history.json()["messages"]
        assert len(messages) == 4  # user, assistant, user, assistant
        assert messages[0]["content"] == "turn one"


def test_chat_validation_error_on_empty_message() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.post("/api/v1/chat", json={"message": ""})
        assert response.status_code == 422


def test_unknown_session_id_falls_back_to_new() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat", json={"message": "hi", "session_id": "missing-id"}
        )
        assert response.status_code == 200
        assert response.json()["session_id"] != "missing-id"


def test_session_endpoints_list_and_get() -> None:
    app = create_app()
    with TestClient(app) as client:
        created = client.post("/api/v1/chat", json={"message": "hi"}).json()
        listing = client.get("/api/v1/sessions")
        assert listing.status_code == 200
        assert created["session_id"] in listing.json()["sessions"]

        detail = client.get(f"/api/v1/sessions/{created['session_id']}")
        assert detail.status_code == 200
        assert len(detail.json()["messages"]) == 2


def test_delete_session_endpoint() -> None:
    app = create_app()
    with TestClient(app) as client:
        created = client.post("/api/v1/chat", json={"message": "hi"}).json()
        sid = created["session_id"]
        deleted = client.delete(f"/api/v1/sessions/{sid}")
        assert deleted.status_code == 200
        listing = client.get("/api/v1/sessions").json()["sessions"]
        assert sid not in listing


def test_error_handler_returns_json() -> None:
    from fastapi import FastAPI, HTTPException

    app = FastAPI()
    register_error_handlers(app)


    @app.get("/boom")
    def boom() -> None:
        raise SessionNotFoundError("custom session missing")

    with TestClient(app) as client:
        response = client.get("/boom")
        assert response.status_code == 404
        body = response.json()
        assert body["error"]["code"] == "session_not_found"
        assert body["error"]["message"] == "custom session missing"