# Tests Directory

## Purpose
This directory contains automated tests for the Simba Mock API. These tests ensure that the API functions correctly and continues to work as expected after any modifications.

## Contents

### test_api.py
This file contains tests for all API endpoints:
- Testing the health endpoint
- Testing API key generation and validation
- Testing the GET and POST `/simbarequest` endpoints
- Testing error handling and authentication

## How to Run Tests

### Prerequisites
Before running tests, make sure you have set up the development environment by running the setup script:
- Linux/macOS: `./setup_dev.sh`
- Windows: `setup_dev.bat`

### Running All Tests
To run all tests:

```bash
# Activate the virtual environment first
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# Then run the tests
pytest
```

### Running Specific Tests
To run a specific test file:

```bash
pytest tests/test_api.py
```

To run a specific test function:

```bash
pytest tests/test_api.py::test_health_endpoint
```

### Test Output
The tests will output information about which tests passed or failed. If all tests pass, you'll see something like:

```
=================== 4 passed in 1.23s ===================
```

If a test fails, the output will include information about what went wrong.

## How to Add New Tests

1. Create a new test file in this directory (e.g., `test_new_feature.py`)
2. Import the necessary modules:
   ```python
   import pytest
   from fastapi.testclient import TestClient
   from app.main import app

   client = TestClient(app)
   ```

3. Define test functions that test specific functionality:
   ```python
   def test_new_feature():
       response = client.get("/your-endpoint")
       assert response.status_code == 200
       # Add more assertions as needed
   ```

4. Run the tests to ensure they work as expected

## Test Coverage

The test suite aims to cover:
- All API endpoints
- Authentication functionality
- Error handling
- Edge cases (invalid IDs, missing files, etc.)

To generate a test coverage report:

```bash
pytest --cov=app tests/
```

This requires the `pytest-cov` package, which can be installed with:

```bash
pip install pytest-cov
```

## Best Practices

1. **Test Isolation**: Each test should be independent and not rely on the state from other tests
2. **Clear Names**: Use descriptive function names that explain what is being tested
3. **Fixtures**: Use pytest fixtures for setup and teardown
4. **Assertions**: Make specific assertions about the expected behavior
5. **Coverage**: Aim for high test coverage, especially for critical functionality

## Technical Details

- Tests use pytest as the testing framework
- FastAPI's TestClient is used to make requests to the API without running a server
- Tests create temporary data when needed and clean up afterwards
