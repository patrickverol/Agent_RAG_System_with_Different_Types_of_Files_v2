#!/bin/bash

# Stock Data Loader Runner Script
# This script can be used to manually run the stock data loader

echo "=== Stock Data Loader Runner ==="
echo "Starting stock data loading process..."

# Check if we're in a Docker environment
if [ -f /.dockerenv ]; then
    echo "Running inside Docker container..."
    python load_stock_data.py
else
    echo "Running outside Docker container..."
    echo "Make sure you have the required environment variables set:"
    echo "- POSTGRES_HOST"
    echo "- POSTGRES_PORT" 
    echo "- POSTGRES_DB"
    echo "- POSTGRES_USER"
    echo "- POSTGRES_PASSWORD"
    echo ""
    echo "You can also run this using Docker Compose:"
    echo "docker-compose run --rm stock_data_loader"
fi 