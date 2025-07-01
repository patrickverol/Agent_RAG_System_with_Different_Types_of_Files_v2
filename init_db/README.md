# Stock Data Loader

This directory contains the stock data loader that fetches real-time stock data using yfinance and saves it to the PostgreSQL database.

## Files

- `load_stock_data.py` - Main Python script that fetches stock data
- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker configuration for the stock data loader
- `run_stock_loader.sh` - Shell script to run the loader manually
- `README.md` - This file

## How it works

1. **Automatic Loading**: When you run `docker-compose up`, the stock data loader will automatically run after the PostgreSQL database is ready
2. **Data Source**: Uses yfinance to fetch real-time stock data for 20 major tech companies
3. **Database Storage**: Saves comprehensive stock data to the `stock_data` table in PostgreSQL

## Stock Symbols

The loader fetches data for these 20 stocks:
- AAPL, MSFT, AMZN, GOOGL, TSLA, META, NFLX, ADBE
- CRM, ORCL, NVDA, INTC, CSCO, IBM, HPQ, DELL
- VMW, NOW, WDAY, SNOW

## Data Fields

The `stock_data` table includes:
- Basic info: symbol, company_name, sector, industry
- Price data: current_price, open_price, day_high, day_low
- Market data: market_cap, volume, avg_volume
- Financial metrics: pe_ratio, dividend_yield
- Change data: change, change_percent
- Additional info: website, description, employees, country

## Manual Execution

### Using Docker Compose
```bash
# Run the stock data loader manually
docker-compose run --rm stock_data_loader

# Or run it and keep the container
docker-compose run stock_data_loader
```

### Using Docker directly
```bash
# Build the image
docker build -t stock-data-loader ./init_db

# Run the container
docker run --rm \
  -e POSTGRES_HOST=your_host \
  -e POSTGRES_PORT=5433 \
  -e POSTGRES_DB=rag_db \
  -e POSTGRES_USER=rag_user \
  -e POSTGRES_PASSWORD=rag_password \
  stock-data-loader
```

### Running locally (outside Docker)
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5433
export POSTGRES_DB=rag_db
export POSTGRES_USER=rag_user
export POSTGRES_PASSWORD=rag_password

# Run the script
python load_stock_data.py
```

## Database Schema

The script creates a `stock_data` table with the following structure:

```sql
CREATE TABLE stock_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL UNIQUE,
    company_name VARCHAR(100),
    current_price DECIMAL(10,2),
    market_cap DECIMAL(15,2),
    volume BIGINT,
    avg_volume BIGINT,
    day_high DECIMAL(10,2),
    day_low DECIMAL(10,2),
    open_price DECIMAL(10,2),
    previous_close DECIMAL(10,2),
    change DECIMAL(10,2),
    change_percent DECIMAL(10,2),
    pe_ratio DECIMAL(10,2),
    dividend_yield DECIMAL(5,2),
    sector VARCHAR(50),
    industry VARCHAR(100),
    country VARCHAR(50),
    website VARCHAR(200),
    description TEXT,
    employees INTEGER,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Logging

The script provides detailed logging:
- Connection status
- Data fetching progress
- Success/failure counts
- Error details

## Error Handling

- Graceful handling of network issues
- Rate limiting protection (0.5s delay between requests)
- Database connection retry logic
- Comprehensive error logging

## Dependencies

- `yfinance==0.2.36` - Yahoo Finance data fetching
- `psycopg2-binary==2.9.9` - PostgreSQL adapter
- `python-dotenv==1.0.1` - Environment variable loading
- `pandas==2.1.4` - Data manipulation
- `requests==2.31.0` - HTTP requests 