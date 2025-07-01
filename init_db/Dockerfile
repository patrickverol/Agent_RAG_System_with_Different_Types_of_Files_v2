FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the stock data loader script
COPY load_stock_data.py .

# Make the script executable
RUN chmod +x load_stock_data.py

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the stock data loader
CMD ["python", "load_stock_data.py"] 