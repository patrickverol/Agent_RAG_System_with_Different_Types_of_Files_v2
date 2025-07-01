"""
API Module
This module implements the FastAPI endpoints for the RAG system.
It handles document queries and provides responses using similarity-based retrieval.
"""

# Import os module
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import FastAPI class from fastapi module to create the API
from fastapi import FastAPI

# Import QdrantVectorStore class from langchain_qdrant module to instantiate the vector database
from langchain_qdrant import QdrantVectorStore

# Import QdrantClient class from qdrant_client module to connect to the vector database
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Import BaseModel class from pydantic module to validate data sent to the API
from pydantic import BaseModel

# Import HuggingFaceEmbeddings class from langchain_huggingface module to generate embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Import similarity calculation
from aux_functions import calcula_similaridade

# Define Item class that inherits from BaseModel
class Item(BaseModel):
    query: str

# Define model name (tokenizer)
model_name = "sentence-transformers/msmarco-bert-base-dot-v5"

# Define model arguments
model_kwargs = {'device': 'cpu'}

# Define encoding arguments
encode_kwargs = {'normalize_embeddings': True}

# Create HuggingFaceEmbeddings instance
hf = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs)

# Create instance to connect to vector database
client = QdrantClient("http://qdrant:6333")

# Define collection name
collection_name = "RAGVectorDB"

# Create collection if it doesn't exist
if not client.collection_exists(collection_name):
    print(f"Creating collection: {collection_name}")
    client.create_collection(
        collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

# Create Qdrant instance to send data to vector database
qdrant = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=hf,
    distance=Distance.COSINE
)

# Initialize FastAPI app
app = FastAPI()

# Define root route with GET method
@app.get("/")
async def root():
    
    # Check Qdrant collection status
    try:
        collection_info = client.get_collection(collection_name)
        points_count = client.count(collection_name).count
        return {
            "message": "RAG Project",
            "collection_status": {
                "name": collection_name,
                "points_count": points_count,
                "status": collection_info.status
            }
        }
    except Exception as e:
        return {
            "message": "RAG Project",
            "error": f"Failed to get collection status: {str(e)}"
        }

# Define /rag_api route with POST method
@app.post("/rag_api")
async def rag_api(item: Item):

    # Get query from item
    query = item.query
    
    # Perform similarity search to get top 10 results
    search_result = qdrant.similarity_search(query = query, k = 10)
    print(f"Similarity search results: {len(search_result)} documents found")  # Debug log
    
    # Initialize results list
    list_res = []
    
    # Process each search result and calculate similarity scores
    for i, res in enumerate(search_result):
        print(f"Processing result {i}: {res.page_content[:100]}...")  # Debug log
        
        # Calculate similarity score between query and content
        similarity_score = calcula_similaridade(query, res.page_content)
        
        # Add result to list with similarity score
        list_res.append({
            "id": res.metadata.get("chunk_id", f"chunk_{i}"),  # Use chunk_id if available
            "path": res.metadata.get("path", ""),
            "content": res.page_content,
            "similarity_score": round(similarity_score * 100, 2),  # Convert to percentage
            "metadata": res.metadata
        })

    # Sort results by similarity score (highest first)
    list_res.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    # Return only the top 5 most relevant results
    top_5_results = list_res[:5]
    
    print(f"Returning top 5 results with similarity scores: {[r['similarity_score'] for r in top_5_results]}")
    
    return {
        "context": top_5_results,
        "query": query,
        "total_results": len(list_res),
        "top_results_count": len(top_5_results)
    }




