#!/usr/bin/env python3
"""
Stock Data Loader
This script fetches stock data for VALE and DIRR using yfinance and saves it to PostgreSQL tables.
"""

import os
import sys
import time
import psycopg2
import yfinance as yf
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'postgres_source'),
    'port': os.getenv('POSTGRES_PORT', '5433'),
    'database': os.getenv('POSTGRES_DB', 'rag_db'),
    'user': os.getenv('POSTGRES_USER', 'rag_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'rag_password')
}

# Stock symbols to fetch data for (Brazilian market)
STOCK_SYMBOLS = ['VALE']

def get_database_connection():
    """Get a connection to the PostgreSQL database."""
    return psycopg2.connect(
        host=DB_CONFIG['host'],
        port=DB_CONFIG['port'],
        database=DB_CONFIG['database'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password']
    )

def create_stock_table(connection, symbol):
    """Create a table for a specific stock if it doesn't exist"""
    try:
        cursor = connection.cursor()
        
        # Create table name for the specific stock
        table_name = f"stock_{symbol.lower()}"
        
        # Create table with the required columns
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            date DATE NOT NULL,
            open DECIMAL(10,2),
            high DECIMAL(10,2),
            low DECIMAL(10,2),
            close DECIMAL(10,2),
            volume BIGINT,
            dividends DECIMAL(10,2),
            stock_splits DECIMAL(10,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        cursor.execute(create_table_query)
        connection.commit()
        logger.info(f"Table {table_name} created successfully")
        cursor.close()
        
    except psycopg2.Error as e:
        logger.error(f"Error creating table for {symbol}: {e}")
        connection.rollback()

def fetch_stock_data(symbol):
    """Fetch stock data for a given symbol using yfinance"""
    try:
        logger.info(f"Fetching data for {symbol}")
        
        # Create Ticker object
        ticker = yf.Ticker(symbol)
        
        # Get 30 days of historical data
        hist = ticker.history(period="30d")
        
        if hist.empty:
            logger.warning(f"No data found for {symbol}")
            return None
        
        # Convert to list of dictionaries
        stock_data_list = []
        for date, row in hist.iterrows():
            stock_data = {
                'date': date.date(),
                'open': float(row['Open']) if not pd.isna(row['Open']) else 0.0,
                'high': float(row['High']) if not pd.isna(row['High']) else 0.0,
                'low': float(row['Low']) if not pd.isna(row['Low']) else 0.0,
                'close': float(row['Close']) if not pd.isna(row['Close']) else 0.0,
                'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0,
                'dividends': float(row['Dividends']) if not pd.isna(row['Dividends']) else 0.0,
                'stock_splits': float(row['Stock Splits']) if not pd.isna(row['Stock Splits']) else 0.0
            }
            stock_data_list.append(stock_data)
        
        logger.info(f"Successfully fetched {len(stock_data_list)} days of data for {symbol}")
        return stock_data_list
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        
        # Try alternative symbol formats for Brazilian stocks
        alternative_symbols = []
        
        # If it has .SA suffix, try without it
        if symbol.endswith('.SA'):
            alternative_symbols.append(symbol.replace('.SA', ''))
        # If it doesn't have .SA suffix, try with it
        else:
            alternative_symbols.append(f"{symbol}.SA")
        
        # Try alternative symbols
        for alt_symbol in alternative_symbols:
            try:
                logger.info(f"Trying alternative symbol: {alt_symbol}")
                ticker = yf.Ticker(alt_symbol)
                hist = ticker.history(period="30d")
                
                if not hist.empty:
                    stock_data_list = []
                    for date, row in hist.iterrows():
                        stock_data = {
                            'date': date.date(),
                            'open': float(row['Open']) if not pd.isna(row['Open']) else 0.0,
                            'high': float(row['High']) if not pd.isna(row['High']) else 0.0,
                            'low': float(row['Low']) if not pd.isna(row['Low']) else 0.0,
                            'close': float(row['Close']) if not pd.isna(row['Close']) else 0.0,
                            'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0,
                            'dividends': float(row['Dividends']) if not pd.isna(row['Dividends']) else 0.0,
                            'stock_splits': float(row['Stock Splits']) if not pd.isna(row['Stock Splits']) else 0.0
                        }
                        stock_data_list.append(stock_data)
                    
                    logger.info(f"Successfully fetched {len(stock_data_list)} days of data for {alt_symbol}")
                    return stock_data_list
            except Exception as alt_e:
                logger.error(f"Alternative symbol {alt_symbol} also failed: {alt_e}")
        
        return None

def save_stock_data(connection, symbol, stock_data_list):
    """Save stock data to the specific stock table"""
    try:
        cursor = connection.cursor()
        
        # Get table name for this stock
        table_name = f"stock_{symbol.lower()}"
        
        # Clear existing data to maintain only 30 days
        cursor.execute(f"DELETE FROM {table_name}")
        
        # Insert data into the specific stock table
        insert_query = f"""
        INSERT INTO {table_name} (
            date, open, high, low, close, volume, dividends, stock_splits
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """
        
        for data in stock_data_list:
            cursor.execute(insert_query, (
                data['date'], data['open'], data['high'], data['low'],
                data['close'], data['volume'], data['dividends'], data['stock_splits']
            ))
        
        connection.commit()
        cursor.close()
        logger.info(f"Successfully saved {len(stock_data_list)} records for {symbol} to table {table_name}")
        
    except psycopg2.Error as e:
        logger.error(f"Error saving data for {symbol}: {e}")
        connection.rollback()

def load_all_stock_data():
    """Main function to load all stock data"""
    logger.info("Starting stock data loading process...")
    
    # Create database connection
    connection = get_database_connection()
    if not connection:
        logger.error("Failed to connect to database. Exiting.")
        return False
    
    try:
        # Create tables for all stocks first
        logger.info("Creating tables for all stocks...")
        for symbol in STOCK_SYMBOLS:
            create_stock_table(connection, symbol)
        
        # Fetch and save data for each symbol
        successful_loads = 0
        failed_loads = 0
        
        for symbol in STOCK_SYMBOLS:
            try:
                # Fetch stock data
                stock_data_list = fetch_stock_data(symbol)
                
                if stock_data_list:
                    # Save to database
                    save_stock_data(connection, symbol, stock_data_list)
                    successful_loads += 1
                else:
                    failed_loads += 1
                
                # Add delay to avoid rate limiting
                time.sleep(5.0)  # 5 seconds between stocks
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                failed_loads += 1
        
        logger.info(f"Stock data loading completed. Successful: {successful_loads}, Failed: {failed_loads}")
        return successful_loads > 0
        
    except Exception as e:
        logger.error(f"Error in load_all_stock_data: {e}")
        return False
    
    finally:
        if connection:
            connection.close()
            logger.info("Database connection closed")

def verify_data_loaded():
    """Verify that data was loaded successfully in all stock tables"""
    connection = get_database_connection()
    if not connection:
        return False
    
    try:
        cursor = connection.cursor()
        total_records = 0
        
        # Check each stock table
        for symbol in STOCK_SYMBOLS:
            table_name = f"stock_{symbol.lower()}"
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            total_records += count
            logger.info(f"Table {table_name}: {count} records")
        
        cursor.close()
        
        logger.info(f"Total records across all stock tables: {total_records}")
        return total_records > 0
        
    except psycopg2.Error as e:
        logger.error(f"Error verifying data: {e}")
        return False
    finally:
        connection.close()

if __name__ == "__main__":
    logger.info("=== Stock Data Loader Started ===")
    
    # Load all stock data
    success = load_all_stock_data()
    
    if success:
        # Verify data was loaded
        if verify_data_loaded():
            logger.info("=== Stock Data Loading Completed Successfully ===")
        else:
            logger.warning("=== Stock Data Loading Completed but No data found in database ===")
    else:
        logger.warning("=== Stock Data Loading Failed - Container will continue running ===")
    
    # Keep container running instead of exiting
    logger.info("=== Container will continue running ===")
    
    # Option to retry loading data periodically
    retry_interval = int(os.getenv('RETRY_INTERVAL_HOURS', '24'))  # Default 24 hours
    logger.info(f"Will retry loading stock data every {retry_interval} hours")
    
    try:
        # Keep the container alive and retry periodically
        while True:
            time.sleep(retry_interval * 3600)  # Convert hours to seconds
            logger.info(f"Retrying stock data loading after {retry_interval} hours...")
            
            # Try to load data again
            retry_success = load_all_stock_data()
            if retry_success:
                logger.info("Retry successful!")
            else:
                logger.warning("Retry failed, will try again later")
                
    except KeyboardInterrupt:
        logger.info("Container stopped by user")
        sys.exit(0) 