# App Directory

## Purpose
This directory contains the main application code for the Simba Mock API for Agent Development. It is structured following FastAPI best practices to provide a clean, maintainable API.

## Contents

### main.py
The entry point for the FastAPI application. This file defines all the API endpoints and their functionality:
- `/health` and `/healthcheck`: Return system health information
- `/api-key`: Generates and returns a new API key
- `/simbarequest` (GET and POST): Retrieve JSON data based on request ID

### core/
This subdirectory contains core functionality that supports the application:
- `config.py`: Configuration settings for the application
- `auth.py`: Authentication utilities for API key generation and validation

### routers/
This subdirectory is reserved for route definitions if you want to modularize the application further. Currently, all routes are defined in `main.py`.

### models/
This subdirectory is reserved for Pydantic models that define request and response schemas. Currently, models are defined directly in `main.py`.

## How to Use

### Adding New Endpoints
To add new endpoints to the API:

1. Define a new route in `main.py` using FastAPI's decorator syntax:
   ```python
   @app.get("/your-endpoint")
   async def your_endpoint_function():
       # Your endpoint logic here
       return {"message": "Your response"}
   ```

2. If you want to modularize your routes, you can create a new file in the `routers` directory:
   ```python
   # app/routers/your_router.py
   from fastapi import APIRouter

   router = APIRouter()

   @router.get("/your-endpoint")
   async def your_endpoint_function():
       # Your endpoint logic here
       return {"message": "Your response"}
   ```

   Then include it in `main.py`:
   ```python
   from app.routers.your_router import router as your_router
   app.include_router(your_router)
   ```

### Working with Authentication
The API uses a simple API key authentication system defined in `core/auth.py`. To use it in new endpoints:

1. Import the authentication function:
   ```python
   from app.core.auth import validate_api_key
   ```

2. Add it as a dependency to your endpoint:
   ```python
   @app.get("/your-protected-endpoint")
   async def your_protected_endpoint(api_key: str = Header(...)):
       if not validate_api_key(api_key):
           raise HTTPException(status_code=401, detail="Invalid API key")
       # Your endpoint logic here
   ```

### Modifying Configuration
Application configuration is managed in `core/config.py`. To modify or add configuration options:

1. Add new settings to the `Settings` class
2. Access settings anywhere in the application using `from app.core.config import settings`

## Technical Details

- The application uses FastAPI for API definition and routing
- Authentication is handled through API keys stored in memory (would be a database in production)
- Configuration is managed through environment variables and a `.env` file
- The API is designed to be easily extensible for new mock data scenarios
- OpenAPI specification is automatically generated from the API definitions

## Dependencies

- FastAPI: Web framework
- Pydantic: Data validation and settings management
- Uvicorn: ASGI server (for running the application)
- Python-dotenv: For loading environment variables from a .env file