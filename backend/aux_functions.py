"""
Dynamic PostgreSQL Connection for RAG System
"""

import psycopg2
import os

# Import hashlib for document ID generation
import hashlib

# Import similarity calculation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
    

def gera_chunk_id(content: str, metadata: dict) -> str:
    """
    Generate a unique chunk ID using MD5 hash of content and metadata.
    
    Args:
        content (str): The chunk content text
        metadata (dict): Metadata dictionary containing path and other info
        
    Returns:
        str: A unique 16-character chunk ID generated from MD5 hash
    """
    
    # Create a string representation of metadata
    metadata_str = str(sorted(metadata.items()))
    
    # Combine content and metadata
    combined = f"{content[:100]}-{metadata_str}"
    
    # Generate MD5 hash of combined string
    hash_object = hashlib.md5(combined.encode('utf-8'))
    
    # Convert hash to hexadecimal
    hash_hex = hash_object.hexdigest()
    
    # Extract first 16 characters for chunk ID
    chunk_id = hash_hex[:16]
    
    return chunk_id


def calcula_similaridade(query: str, context: str) -> float:
    """
    Calculate similarity score between query and context using TF-IDF and cosine similarity.
    
    Args:
        query (str): The user query
        context (str): The context text to compare against
        
    Returns:
        float: Similarity score between 0 and 1 (1 being most similar)
    """
    try:
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        
        # Fit and transform the documents
        documents = [query, context]
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        
        # Return the similarity score
        similarity_score = float(similarity_matrix[0][0])
        
        return similarity_score
        
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0


def verifica_chunk_existente(chunk_id: str, qdrant_client) -> bool:
    """
    Check if a chunk with the given ID already exists in Qdrant database.
    
    Args:
        chunk_id (str): The chunk ID to check
        qdrant_client: Qdrant client instance
        
    Returns:
        bool: True if chunk exists, False otherwise
    """
    try:
        # Use scroll to check for existing chunks with the same chunk_id
        scroll_result = qdrant_client.scroll(
            collection_name="RAGVectorDB",
            scroll_filter={"must": [{"key": "chunk_id", "match": {"value": chunk_id}}]},
            limit=1
        )
        
        # If we find any results, the chunk exists
        return len(scroll_result[0]) > 0
        
    except Exception as e:
        print(f"Error checking chunk existence: {e}")
        return False
