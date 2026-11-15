"""The port is closed: no session, no data.

Before this, every /api route answered anyone who could reach the port —
submit a job, list clips, delete media. The UI had a login screen and a
`RequireAuth` wrapper, but no backend ever checked the session, so the gate was
cosmetic. These tests pin the backend half:

* anonymous calls get 401 (except /api/auth/* and the slim /api/health),
* the first sign-up creates the owner and sets an HttpOnly cookie,
* a second sign-up is refused,
* wrong password → 401, right password → 200 + working cookie,
* sign-out invalidates the cookie server-side,
* the local automation token still works for tools, a random one does not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import auth


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("CLIPFORGE_DATA_DIR", str(tmp_path))
    auth.reset_for_tests()
    from app.main import app  # imported after the env var so auth writes to tmp_path

    with TestClient(app) as c:
        yield c
    auth.reset_for_tests()


def test_anonymous_api_is_401(client: TestClient) -> None:
    for path in ("/api/jobs", "/api/clips", "/api/storage/unused"):
        res = client.get(path)
        assert res.status_code == 401, path
        assert res.json().get("authRequired") is True


def test_health_is_public_but_slim(client: TestClient) -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["authRequired"] is True
    assert "worker" not in body, "anonymous health must not leak pipeline detail"


def test_first_signup_owns_the_instance(client: TestClient) -> None:
    res = client.post(
        "/api/auth/signup",
        json={"email": "owner@example.com", "password": "hunter2hunter2", "name": "Owner"},
    )
    assert res.status_code == 200, res.text
    assert res.json()["user"]["email"] == "owner@example.com"
    cookie = res.headers.get("set-cookie", "")
    assert auth.COOKIE_NAME in cookie
    assert "HttpOnly" in cookie

    # The cookie opened the API.
    assert client.get("/api/jobs").status_code == 200
    me = client.get("/api/auth/me").json()
    assert me["user"]["email"] == "owner@example.com"
    assert me["signupOpen"] is False

    # A second account cannot be created by a stranger.
    anon = TestClient(client.app)
    res = anon.post(
        "/api/auth/signup",
        json={"email": "intruder@example.com", "password": "hunter2hunter2"},
    )
    assert res.status_code == 403


def test_signin_signout_cycle(client: TestClient) -> None:
    client.post(
        "/api/auth/signup",
        json={"email": "owner@example.com", "password": "hunter2hunter2"},
    )
    assert client.post("/api/auth/signout").status_code == 200
    assert client.get("/api/jobs").status_code == 401
    assert client.get("/api/auth/me").json()["user"] is None

    bad = client.post(
        "/api/auth/signin", json={"email": "owner@example.com", "password": "wrong-password"}
    )
    assert bad.status_code == 401

    good = client.post(
        "/api/auth/signin", json={"email": "owner@example.com", "password": "hunter2hunter2"}
    )
    assert good.status_code == 200
    assert client.get("/api/jobs").status_code == 200


def test_automation_token_opens_the_api_but_a_guess_does_not(client: TestClient, tmp_path: Path) -> None:
    token = auth.automation_token()
    assert len(token) >= 32
    assert (tmp_path / "auth" / "automation.token").exists()

    assert client.get("/api/jobs", headers={"X-ClipForge-Token": token}).status_code == 200
    assert client.get("/api/jobs", headers={"X-ClipForge-Token": "nope"}).status_code == 401


def test_password_is_not_stored_in_the_clear(client: TestClient, tmp_path: Path) -> None:
    client.post(
        "/api/auth/signup",
        json={"email": "owner@example.com", "password": "hunter2hunter2"},
    )
    raw = (tmp_path / "auth" / "state.json").read_text(encoding="utf-8")
    assert "hunter2hunter2" not in raw
    assert "scrypt$" in raw
