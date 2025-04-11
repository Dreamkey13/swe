"""
Tests for the Simba Mock API for Agent Development.
"""

import json
import os
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings

# Create test client
client = TestClient(app)

# Mock data for testing
test_data = {
    "message": "This is test data",
    "status": "success",
    "data": {"id": 999, "name": "Test Item", "attributes": ["test"]},
}


@pytest.fixture(scope="module")
def api_key():
    """Get an API key for testing."""
    response = client.get("/api-key")
    assert response.status_code == 200
    return response.json()["api_key"]


@pytest.fixture(scope="module")
def setup_test_data():
    """Set up test data file."""
    # Create a test data file
    test_file_path = os.path.join(settings.DATA_DIR, "request_999.json")

    # Write test data to file
    with open(test_file_path, "w") as f:
        json.dump(test_data, f)

    yield

    # Clean up
    if os.path.exists(test_file_path):
        os.remove(test_file_path)


def test_health_endpoint():
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    # Check that all required fields are present
    assert "hostname" in data
    assert "ip_address" in data
    assert "local_time" in data
    assert "utc_time" in data
    assert "uptime_seconds" in data

    # Validate types
    assert isinstance(data["hostname"], str)
    assert isinstance(data["ip_address"], str)
    assert isinstance(data["local_time"], str)
    assert isinstance(data["utc_time"], str)
    assert isinstance(data["uptime_seconds"], (int, float))


def test_api_key_endpoint():
    """Test the API key endpoint."""
    response = client.get("/api-key")
    assert response.status_code == 200
    data = response.json()

    # Check that API key is present
    assert "api_key" in data
    assert isinstance(data["api_key"], str)
    assert len(data["api_key"]) == settings.API_KEY_LENGTH


def test_get_simba_request(api_key, setup_test_data):
    """Test the GET simba request endpoint."""
    # Test with valid request ID
    response = client.get("/simbarequest?request_id=999", headers={"api_key": api_key})
    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert data["request_id"] == 999
    assert data["data"] == test_data

    # Test with invalid request ID
    response = client.get("/simbarequest?request_id=9999", headers={"api_key": api_key})
    assert response.status_code == 404

    # Test with invalid API key
    response = client.get(
        "/simbarequest?request_id=999", headers={"api_key": "invalid_key"}
    )
    assert response.status_code == 401


def test_post_simba_request(api_key, setup_test_data):
    """Test the POST simba request endpoint."""
    # Test with valid request ID
    response = client.post("/simbarequest?request_id=999", headers={"api_key": api_key})
    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert data["request_id"] == 999
    assert data["data"] == test_data

    # Test with invalid request ID
    response = client.post(
        "/simbarequest?request_id=9999", headers={"api_key": api_key}
    )
    assert response.status_code == 404

    # Test with invalid API key
    response = client.post(
        "/simbarequest?request_id=999", headers={"api_key": "invalid_key"}
    )
    assert response.status_code == 401
