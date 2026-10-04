"""API tests for GET /health and the service's endpoint surface."""

import psycopg


def test_health_reports_the_served_model_when_the_database_answers(client, db, serve):
    """Healthy means: a model is loaded and a recommendation could be logged right now."""
    served = serve((0.95, 0.03, 0.01, 0.005, 0.005))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_name": served.name,
        "model_version": served.version,
        "elevated_threshold": served.threshold,
    }
    assert db.executed == ["SELECT 1"]


def test_health_is_503_when_the_database_is_unreachable(client, db, serve):
    """No database means no logging, so the service reports itself unhealthy."""
    serve((0.95, 0.03, 0.01, 0.005, 0.005))
    db.connect_error = psycopg.OperationalError("connection refused")

    response = client.get("/health")

    assert response.status_code == 503
    assert "cannot be logged" in response.json()["detail"]


def test_service_exposes_only_health_and_predict(client):
    """No endpoint exists that could trigger an action: the surface is two routes."""
    paths = client.get("/openapi.json").json()["paths"]

    assert set(paths) == {"/health", "/predict"}
    assert set(paths["/health"]) == {"get"}
    assert set(paths["/predict"]) == {"post"}