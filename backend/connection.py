"""
Dynamic PostgreSQL Connection for RAG System
"""

import psycopg2
import os


def get_database_connection():
    """Get a connection to the PostgreSQL database."""
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres_source'),
        port=os.getenv('POSTGRES_PORT', '5433'),
        database=os.getenv('POSTGRES_DB', 'rag_db'),
        user=os.getenv('POSTGRES_USER', 'rag_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'rag_password')
    )


def get_all_tables_data():
    """Get data from all tables in all schemas."""
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        
        # Get all schemas and tables
        cursor.execute("""
            SELECT 
                schemaname,
                tablename
            FROM pg_tables 
            WHERE schemaname NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
            ORDER BY schemaname, tablename
        """)
        
        schemas_tables = cursor.fetchall()
        
        all_data = {}
        
        for schema_name, table_name in schemas_tables:
            try:
                # Get data from each table
                cursor.execute(f"SELECT * FROM {schema_name}.{table_name} LIMIT 500")
                rows = cursor.fetchall()
                
                if rows:
                    # Get column names
                    columns = [desc[0] for desc in cursor.description]
                    
                    # Convert to list of dictionaries
                    table_data = []
                    for row in rows:
                        table_data.append(dict(zip(columns, row)))
                    
                    all_data[f"{schema_name}.{table_name}"] = {
                        'schema': schema_name,
                        'table': table_name,
                        'data': table_data,
                        'columns': columns
                    }
                    
                    print(f"Found {len(table_data)} rows in {schema_name}.{table_name}")
                
            except Exception as e:
                print(f"Error reading table {schema_name}.{table_name}: {e}")
                continue
        
        cursor.close()
        conn.close()
        
        return all_data
        
    except Exception as e:
        print(f"Database connection error: {e}")
        return {}


def get_financial_data():
    """Get financial data from PostgreSQL database (backward compatibility)."""
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM financial_data LIMIT 500")
        rows = cursor.fetchall()
        
        # Get column names
        columns = [desc[0] for desc in cursor.description]
        
        cursor.close()
        conn.close()
        
        # Convert to list of dictionaries
        data = []
        for row in rows:
            data.append(dict(zip(columns, row)))
        
        return data
        
    except Exception as e:
        print(f"Database connection error: {e}")
        return [] 