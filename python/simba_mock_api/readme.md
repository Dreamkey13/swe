# Simba Mock API for Agent Development

## Project Overview
This is a Python-based web API that serves as a mock environment for agent development. The API exposes endpoints that return predefined JSON responses, enabling developers to test their agents against consistent mock data.

## Technical Requirements
- Python 3.8 or higher
- FastAPI framework
- Dependencies are listed in `requirements.txt`

## Setup and Installation

### Linux/macOS
1. Clone the repository
2. Run the setup script:
   ```bash
   ./setup_dev.sh
   ```
3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
4. Start the API:
   ```bash
   uvicorn app.main:app --reload
   ```

### Windows
1. Clone the repository
2. Run the setup script:
   ```cmd
   setup_dev.bat
   ```
3. Activate the virtual environment:
   ```cmd
   venv\Scripts\activate
   ```
4. Start the API:
   ```cmd
   uvicorn app.main:app --reload
   ```

## API Endpoints

### GET /health or /healthcheck
Returns system health information.

**Request:**
```bash
curl -X GET http://localhost:8000/health
```

**Response:**
```json
{
  "hostname": "computer-name",
  "ip_address": "192.168.1.100",
  "local_time": "2025-04-10 12:34:56",
  "utc_time": "2025-04-10 04:34:56",
  "uptime_seconds": 1234.56
}
```

### GET /api-key
Generates and returns a new API key that can be used for authentication.

**Request:**
```bash
curl -X GET http://localhost:8000/api-key
```

**Response:**
```json
{
  "api_key": "abcdef1234567890abcdef1234567890",
  "expires_at": null
}
```

### GET /simbarequest
Retrieves JSON data based on the provided request ID.

**Request:**
```bash
curl -X GET "http://localhost:8000/simbarequest?request_id=1" -H "api_key: YOUR_API_KEY"
```

**Response:**
```json
{
  "request_id": 1,
  "data": {
    "message": "This is sample response data for request ID 1",
    "status": "success",
    "timestamp": "2025-04-10T12:00:00Z",
    "data": {
      "id": 1,
      "name": "Sample Item 1",
      "description": "This is a sample description",
      "attributes": ["fast", "reliable", "secure"],
      "metadata": {
        "version": "1.0.0",
        "created_by": "Simba Mock API"
      }
    }
  }
}
```

### POST /simbarequest
Retrieves JSON data based on the provided request ID (same as GET but using POST method).

**Request:**
```bash
curl -X POST "http://localhost:8000/simbarequest?request_id=1" -H "api_key: YOUR_API_KEY"
```

**Response:** 
Same as GET /simbarequest

## Authentication
All `/simbarequest` endpoints require an API key for authentication. The API key should be provided in the `api_key` header. You can obtain an API key from the `/api-key` endpoint.

## Adding New Mock Responses
To add a new mock response:

1. Create a new JSON file in the `data` directory
2. Name the file `request_X.json` where X is the request ID number
3. Format the JSON file according to your requirements
4. The file will be automatically accessible via the `/simbarequest` endpoint with the corresponding request ID

For more details on JSON file format and organization, see the [data folder README](data/README.md).

## Configuration Options
The API can be configured using environment variables in a `.env` file. Copy the `.env.example` file to `.env` and modify as needed.

Available configuration options:
- `HOST`: Host address to bind the server to (default: 0.0.0.0)
- `PORT`: Port to run the server on (default: 8000)
- `API_KEY_LENGTH`: Length of generated API keys (default: 32)
- `DATA_DIR`: Path to the directory containing JSON response files (default: data)

## Documentation
- Swagger UI is available at http://localhost:8000/docs
- OpenAPI specification is available at http://localhost:8000/openapi.json
- Additional documentation is available in the README files in each folder:
  - [App folder documentation](app/README.md)
  - [Data folder documentation](data/README.md)
  - [Tests folder documentation](tests/README.md)

## Running Tests
To run the tests:

```bash
pytest tests/
```

For more information on testing, see the [tests folder README](tests/README.md).
