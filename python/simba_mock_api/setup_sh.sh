#!/bin/bash
# Setup script for development environment on Linux/macOS

# Exit on error
set -e

echo "Setting up development environment for Simba Mock API..."

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}')
major_version=$(echo $python_version | cut -d. -f1)
minor_version=$(echo $python_version | cut -d. -f2)

if [ "$major_version" -lt 3 ] || ([ "$major_version" -eq 3 ] && [ "$minor_version" -lt 8 ]); then
    echo "Error: Python 3.8 or higher is required. Found Python $python_version"
    exit 1
fi

echo "Python $python_version found."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create example .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
fi

# Create data directory if it doesn't exist
if [ ! -d "data" ]; then
    echo "Creating data directory..."
    mkdir -p data
fi

echo ""
echo "Setup complete! You can now activate the virtual environment with:"
echo "source .venv/bin/activate"
echo ""
echo "Then start the API with:"
echo "uvicorn app.main:app --reload"
echo ""
echo "The API will be available at http://localhost:8000"
echo "Swagger UI will be available at http://localhost:8000/docs"
