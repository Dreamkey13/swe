"""
Simba Mock API for Agent Development - Main Application
"""

import os
import socket
import time
import datetime
from typing import Dict, Optional, Union
from fastapi import FastAPI, Depends, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field

# Import local modules
from app.core.config import settings
from app.core.auth import validate_api_key, generate_api_key

# Global variables
start_time = time.time()
app = FastAPI(
    title="Mock API for Agent Development",
    description="A FastAPI-based service that provides mock data for agent development",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class HealthResponse(BaseModel):
    hostname: str
    ip_address: str
    local_time: str
    utc_time: str
    uptime_seconds: float


class ApiKeyResponse(BaseModel):
    api_key: str
    expires_at: Optional[str] = None


class SimbaResponse(BaseModel):
    request_id: int
    data: Dict


# Endpoints
@app.get("/health", response_model=HealthResponse)
@app.get("/healthcheck", response_model=HealthResponse)
async def health_check():
    """
    Returns system health information including hostname, IP address,
    local and UTC time, and service uptime.
    """
    hostname = socket.gethostname()
    try:
        ip_address = socket.gethostbyname(hostname)
    except:
        ip_address = "127.0.0.1"  # Default to localhost if we can't get IP

    local_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    utc_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    uptime = time.time() - start_time

    return {
        "hostname": hostname,
        "ip_address": ip_address,
        "local_time": local_time,
        "utc_time": utc_time,
        "uptime_seconds": uptime,
    }


@app.get("/api-key", response_model=ApiKeyResponse)
async def get_api_key():
    """
    Generates and returns a new API key that can be used for authentication.
    """
    api_key = generate_api_key()
    return {
        "api_key": api_key,
        "expires_at": None,  # In a production system, we might set an expiration
    }


async def get_simba_data(request_id: int):
    """
    Helper function to get simba data based on request ID
    """
    # Check if request_id is valid
    if request_id < 0:
        raise HTTPException(
            status_code=400, detail="requestId must be a positive integer"
        )

    # Get data file path
    data_path = os.path.join(settings.DATA_DIR, f"request_{request_id}.json")

    # Check if file exists
    if not os.path.exists(data_path):
        raise HTTPException(
            status_code=404, detail=f"No data found for requestId: {request_id}"
        )

    # Read JSON file
    try:
        import json

        with open(data_path, "r") as file:
            data = json.load(file)
        return data
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error reading data file: {str(e)}"
        )


@app.get("/simbarequest", response_model=SimbaResponse)
async def get_simba_request(
    request_id: int = Query(
        ..., description="The numeric ID of the request to retrieve"
    ),
    api_key: str = Header(..., description="API key for authentication"),
):
    """
    Retrieves JSON data based on the provided request ID.
    Requires a valid API key in the headers.
    """
    if not validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    data = await get_simba_data(request_id)
    return {"request_id": request_id, "data": data}


@app.post("/simbarequest", response_model=SimbaResponse)
async def post_simba_request(
    request_id: int = Query(
        ..., description="The numeric ID of the request to retrieve"
    ),
    api_key: str = Header(..., description="API key for authentication"),
):
    """
    Retrieves JSON data based on the provided request ID.
    Requires a valid API key in the headers.
    POST version is functionally identical to the GET version.
    """
    if not validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    data = await get_simba_data(request_id)
    return {"request_id": request_id, "data": data}


# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Simba Mock API for Agent Development",
        version="1.0.0",
        description="A FastAPI-based service that provides mock data for agent development",
        routes=app.routes,
    )

    # Add security scheme for API Key
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "api_key"}
    }

    # Apply security to all operations
    for path in openapi_schema["paths"].values():
        for operation in path.values():
            if operation.get("operationId") not in ["health_check", "get_api_key"]:
                operation["security"] = [{"ApiKeyAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
