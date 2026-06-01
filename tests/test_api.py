from __future__ import annotations

from fastapi.testclient import TestClient

from slonik.serve import api as api_module
from slonik.serve.api import GenerateRequest


def test_request_schema_alias():
    req = GenerateRequest(schema="CREATE TABLE t(id INT);", question="q?")
    assert req.schema_ == "CREATE TABLE t(id INT);"


def test_health_endpoint(monkeypatch):
    api_module.CONFIG = {"model": {"served_name": "slonik-7b", "path": ""}}
    client = TestClient(api_module.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
