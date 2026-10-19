"""Smoke tests for the dashboard and the preserved prediction flow."""

from app import app


def test_dashboard_route_and_payload():
    app.config.update(TESTING=True)
    client = app.test_client()

    page = client.get("/")
    payload = client.get("/api/dashboard")

    assert page.status_code == 200
    assert b"Interactive slicers" in page.data
    assert payload.status_code == 200
    data = payload.get_json()
    assert data["metadata"]["row_count"] == 10000
    assert len(data["model"]["model_comparison"]) == 7
    assert data["model"]["confusion_matrices"]


def test_prediction_api_rejects_invalid_input():
    app.config.update(TESTING=True)
    response = app.test_client().post("/api/predict", json={"age": 12})

    assert response.status_code == 400
    assert "Please provide" in response.get_json()["error"]


def test_prediction_api_scores_valid_profile():
    app.config.update(TESTING=True)
    response = app.test_client().post(
        "/api/predict",
        json={
            "age": 42,
            "gender": "Female",
            "tenure": 5,
            "CreditScore": 650,
            "IsActiveMember": 1,
            "Geography": "France",
            "NumOfProducts": "One",
        },
    )

    assert response.status_code == 200
    assert response.get_json()["prediction"] in (0, 1)
