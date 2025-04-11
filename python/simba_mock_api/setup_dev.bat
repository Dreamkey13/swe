@echo off
REM Setup script for development environment on Windows

echo Setting up development environment for Simba Mock API...

REM Check Python version
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    exit /b 1
)

for /f "tokens=2" %%I in ('python --version 2^>^&1') do set PYTHON_VERSION=%%I
echo Python %PYTHON_VERSION% found.

REM Create virtual environment
echo Creating virtual environment...
python -m venv .venv

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Create example .env file if it doesn't exist
if not exist .env (
    echo Creating .env file from example...
    copy .env.example .env
)

REM Create data directory if it doesn't exist
if not exist data (
    echo Creating data directory...
    mkdir data
)

echo.
echo Setup complete! You can now activate the virtual environment with:
echo .venv\Scripts\activate
echo.
echo Then start the API with:
echo uvicorn app.main:app --reload
echo.
echo The API will be available at http://localhost:8000
echo Swagger UI will be available at http://localhost:8000/docs
