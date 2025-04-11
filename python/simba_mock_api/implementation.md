# Simba Mock API Implementation Guide

## Project Overview
This repository contains a complete implementation of the Simba Mock API for Agent Development as specified in the requirements. The API serves predefined JSON responses for agent testing, includes authentication, and provides health monitoring.

## Repository Structure
```
simba_mock_api/
├── app/                  # Application code
├── data/                 # JSON data files
├── tests/                # Test cases
├── setup_dev.sh          # Linux/macOS setup script
├── setup_dev.bat         # Windows setup script
├── requirements.txt      # Project dependencies
├── README.md             # Main documentation
└── .env.example          # Example environment variables
```

## Key Features
- FastAPI-based web service with Swagger UI
- GET and POST endpoints for retrieving mock data
- API key authentication system
- Health monitoring endpoint
- Comprehensive test suite

## Running the Project

### Setup
1. Clone this repository
2. Run the appropriate setup script:
   - Linux/macOS: `./setup_dev.sh`
   - Windows: `setup_dev.bat`

### Starting the API
1. Activate the virtual environment:
   - Linux/macOS: `source .venv/bin/activate`
   - Windows: `.venv\Scripts\activate`
2. Start the server: `uvicorn app.main:app --reload`
3. Access the API at http://localhost:8000
4. Access Swagger UI at http://localhost:8000/docs

### Using the API
1. Get an API key from the `/api-key` endpoint
2. Use the key in subsequent requests to the `/simbarequest` endpoint
3. Check system health with the `/health` endpoint

## Testing
Run the automated tests with:
```bash
pytest tests/
```

## Extending the Project
- Add new JSON files to the `data` directory to create new mock responses
- Modify the endpoints in `app/main.py` to add new functionality
- Add new configuration options in `app/core/config.py`

## Documentation
Comprehensive documentation is available in the README files throughout the project:
- [Main README](README.md) - Overview and usage instructions
- [App README](app/README.md) - Application structure and extension
- [Data README](data/README.md) - JSON file format and organization
- [Tests README](tests/README.md) - Running and extending tests
