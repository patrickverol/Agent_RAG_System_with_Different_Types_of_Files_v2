<h1 align="center">
    RAG System with Different Types of Files - V2.0
</h1>

<br>
    <div align="center">
        <a><img src="https://github.com/user-attachments/assets/57d1cbdd-3562-411b-b1c6-9f2c2aeebb41"></a> 
    </div>
</br>

<div align="center">
    <a href = "https://www.python.org/" target="_blank"><img src="https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white" target="_blank"></a>
    <a href = "https://docs.docker.com/"><img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white" target="_blank"></a>
    <a href = "https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-009688.svg?style=for-the-badge&logo=FastAPI&logoColor=white" target="_blank"></a>
    <a href = "https://qdrant.tech/"><img src="https://img.shields.io/badge/Qdrant-FF4B4B.svg?style=for-the-badge&logo=Qdrant&logoColor=white" target="_blank"></a>
    <a href = "https://docs.streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white" target="_blank"></a>
    <a href = "https://www.postgresql.org/"><img src="https://img.shields.io/badge/PostgreSQL-336791.svg?style=for-the-badge&logo=PostgreSQL&logoColor=white" target="_blank"></a>
    <a href = "https://grafana.com/"><img src="https://img.shields.io/badge/Grafana-F46800.svg?style=for-the-badge&logo=Grafana&logoColor=white" target="_blank"></a>
    <a href = "https://langchain.com/"><img src="https://img.shields.io/badge/LangChain-00FF00.svg?style=for-the-badge&logo=LangChain&logoColor=white" target="_blank"></a>
    <a href = "https://langsmith.langchain.com/"><img src="https://img.shields.io/badge/LangSmith-FF6B6B.svg?style=for-the-badge&logo=LangSmith&logoColor=white" target="_blank"></a>
</div> 

## About the project

This project implements an **Agentic Retrieval-Augmented Generation (RAG) system** that can handle different types of files and intelligently route queries between RAG and web search based on similarity scores. The system uses FastAPI for the backend, Qdrant as a vector database, Streamlit for the frontend interface, and includes comprehensive monitoring capabilities with Grafana, LangSmith, and LogFire.

### 🚀 Key Features in V2.0

- **Multi-Format Document Support**: Handles PDF, DOCX, PPTX, TXT, **XLSX**, **CSV**, and **SQL** files
- **Agentic RAG System**: Intelligent decision-making between RAG and web search based on similarity thresholds
- **Web Search Integration**: Automatic fallback to web search when RAG context is insufficient
- **Graphics Generation**: Creates charts and visualizations based on retrieved context
- **Advanced Observability**: Comprehensive logging and tracing with LangSmith and LogFire
- **Similarity-Based Routing**: Uses embedding-based similarity scoring for optimal source selection
- **Real-time Decision Tracking**: Shows users whether answers come from RAG or web sources

## Streamlit interface

<br>
    <div align="center">
        <a><img src="https://github.com/user-attachments/assets/4549fd58-d5a9-4cee-a5f5-cb4efb027d06"></a> 
    </div>
</br>

## Grafana dashboard

<br>
    <div align="center">
        <a><img src="https://github.com/user-attachments/assets/a75685db-0e83-44e2-9cbd-4aee55a38c6e"></a> 
    </div>
</br>

## LangSmith

<br>
    <div align="center">
        <a><img src="https://github.com/user-attachments/assets/5bff4edd-721c-4d3b-bba6-5ec6dafe7e42"></a> 
    </div>
</br>

## Logfire

<br>
    <div align="center">
        <a><img src="https://github.com/user-attachments/assets/e4224525-f380-4e45-b3d6-237c9abc8fa9"></a> 
    </div>
</br>

## Architecture Overview

The system is designed with a microservices architecture, divided into several components for better scalability and maintainability:

### 1. **Backend Environment (FastAPI)**
   - Handles document processing and embedding for multiple file formats
   - Manages interactions with Qdrant vector database
   - Provides REST API endpoints for RAG operations
   - Processes XLSX, CSV, and SQL files with structured data handling
   - Advantages:
     - Fast and efficient API handling
     - Async support for better performance
     - Multi-format document processing
     - Scalable document processing

### 2. **Frontend Environment (Streamlit)**
   - Provides user interface for document upload and query
   - Displays search results with source attribution (RAG/Web)
   - Shows agent decision-making process
   - Supports graphics generation and visualization
   - Advantages:
     - User-friendly interface with decision transparency
     - Real-time updates and graphics display
     - Easy to customize and extend
     - Responsive design

### 3. **Agent System (LangGraph)**
   - Implements intelligent routing between RAG and web search
   - Uses similarity-based decision making (70% threshold)
   - Supports graphics generation based on context
   - Provides comprehensive observability
   - Advantages:
     - Intelligent source selection
     - Fallback mechanisms for better coverage
     - Visual data representation
     - Full traceability of decisions

### 4. **Storage Environment**
   - Qdrant for vector storage with unique chunk IDs
   - PostgreSQL for metadata storage and evaluation
   - Document storage service for file management
   - Advantages:
     - Efficient vector similarity search
     - Reliable metadata management
     - Flexible document storage options
     - Scalable architecture

### 5. **Observability Stack**
   - **LangSmith**: LLM tracing and debugging
   - **LogFire**: Structured logging and monitoring
   - **Grafana**: Metrics visualization and dashboards
   - Advantages:
     - Comprehensive system monitoring
     - Debugging capabilities for LLM chains
     - Performance insights
     - Error tracking and alerting

## Project Structure

```
├── backend/                                                # FastAPI backend
│   ├── api.py                                             # API endpoints
│   ├── aux_functions.py                                   # Utility functions
│   ├── rag.py                                             # RAG processing (XLSX/CSV/SQL support)
│   ├── storage.py                                         # Storage management
│   ├── start_api.py                                       # API startup
│   ├── backend.Dockerfile                                 # Container configuration
│   └── requirements.txt                                   # Python dependencies
├── frontend/                                               # Streamlit frontend
│   ├── agent.py                                           # LangGraph agent implementation
│   ├── web_app.py                                         # Streamlit interface
│   ├── connection.py                                      # Backend connection
│   ├── llm.py                                             # LLM configuration
│   ├── qdrant.py                                          # Vector DB operations
│   ├── storage.py                                         # Storage operations
│   ├── frontend.Dockerfile                                # Container configuration
│   └── requirements.txt                                   # Python dependencies
├── document_storage/                                       # Document storage service
│   ├── app.py                                             # Storage application
│   ├── Dockerfile                                         # Container configuration
│   └── requirements.txt                                   # Python dependencies
├── init_db/                                               # Database initialization
│   ├── load_stock_data.py                                 # Stock data loader
│   ├── Dockerfile                                         # Container configuration
│   └── requirements.txt                                   # Python dependencies
├── observability/                                         # Monitoring and observability
│   └── dashboard.json                                     # Grafana dashboard configuration
├── docker-compose.yaml                                    # Infrastructure services
└── README.md                                              # Project documentation
```

---

## Accessing the Services

1. **Backend API**
   - URL: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

2. **Frontend Interface**
   - URL: http://localhost:8501

3. **Grafana Dashboard**
   - URL: http://localhost:3000
   - Default credentials:
     - Username: admin
     - Password: admin

4. **LangSmith Tracing** (Optional)
   - Access through LangSmith dashboard with API key
   - Provides detailed LLM chain tracing and debugging

5. **Logfire** (Optional)
   - Access through Logfire dashboard with API key
   - Provides structured logging, distributed tracing, and real-time monitoring
   - Tracks agent decisions, similarity scores, and performance metrics
---

## Requirements

- Docker
- Docker Compose
- Python 3.9+
- API Keys (optional but recommended):
  - GROQ_API_KEY (for LLM operations)
  - LANGSMITH_API_KEY (for tracing)
  - LOGFIRE_TOKEN (for logging)

---

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Set up environment variables**
   Create a `.env` file or set environment variables in docker-compose:
   ```bash
   # Required for LLM operations
   GROQ_API_KEY=your_groq_api_key
   
   # Optional for observability
   LANGSMITH_API_KEY=your_langsmith_api_key
   LANGCHAIN_API_KEY=your_langchain_api_key
   LOGFIRE_TOKEN=your_logfire_token
   
   # Storage configuration
   STORAGE_TYPE=local
   DOCUMENTS_PATH=/app/documents
   ```

3. **Start the services**
   ```bash
   docker-compose up -d
   ```

4. **Wait for all services to start**
   The system includes the following services:
   - Backend (FastAPI)
   - Frontend (Streamlit)
   - Qdrant (Vector Database)
   - PostgreSQL (Metadata Storage)
   - Document Storage
   - Grafana (Monitoring)
   - Stock Data Loader (Database Initialization)

5. **Populate the Vector Database**
   Go to the backend container terminal and run:
   ```bash
   python rag.py
   ```
   This script will process documents from the document_storage container, apply embeddings, and send the data to the Qdrant vector database.

6. **Access the Web App**
   - Open the frontend interface at http://localhost:8501
   - Use the search functionality to query your documents
   - The system will automatically decide between RAG and web search
   - View the agent's decision (RAG/Web) and generated graphics
   - Example:
   <br>
    <div align="center">
        <a><img src="https://github.com/user-attachments/assets/4549fd58-d5a9-4cee-a5f5-cb4efb027d06"></a> 
    </div>
   </br>

7. **Access the Grafana dashboard**
   - Monitor system performance through Grafana dashboard
   - Open the grafana interface at http://localhost:3000
   - Username: `admin`
   - Password: `admin` (*The browser will show a message to change the password, just ignore that and write the same password again*)
   - Go to `Connections` and search for `PostgreSQL`
   - Click in the connection and after that click on `Add new data source`
   - Make the configurations like below:
      - Host URL: `<your_postgre_container_hostname>:5432` (*You can see your postgres hostname running the command below in the terminal*)
      ```bash
      docker exec rag_postgres_evaluate hostname
      ```
      - Database name: `admin`
      - Username: `admin`
      - Password: `admin`
      - TLS/SSL Mode: `disable`
   - Click on `Save & test`
   - Click `Dashboard` > `New` > `Import`
   - Upload the dashboard saved in the folder `observability`
   - After loading the dashboard, in the first time of the visualization you need to update each panel to run the query
   - For each visualization:
      - Click on the `three dots in the top right corner` > `Edit` > `Run query`
   - After the first update, the visualizations will be update automatically according to the refresh interval configured
   <br>
    <div align="center">
        <a><img src="https://github.com/user-attachments/assets/a75685db-0e83-44e2-9cbd-4aee55a38c6e"></a> 
    </div>
   </br>

8. **Access LangSmith** (Optional)
   - Go to [LangSmith Dashboard](https://smith.langchain.com/)
   - Sign in with your LangSmith account or create a new one
   - Set up your `LANGSMITH_API_KEY` in the environment variables
   - View detailed traces of LLM chains, agent decisions, and performance metrics
   - Debug and analyze the agent's decision-making process
   - Monitor similarity scores and routing decisions in real-time
   <br>
    <div align="center">
        <a><img src="https://github.com/user-attachments/assets/5bff4edd-721c-4d3b-bba6-5ec6dafe7e42"></a> 
    </div>
   </br>

9. **Access LogFire** (Optional)
   - Go to [LogFire Dashboard](https://cloud.logfire.sh/)
   - Sign in with your LogFire account or create a new one
   - Set up your `LOGFIRE_TOKEN` in the environment variables
   - View structured logs with rich metadata and context
   - Monitor distributed traces across microservices
   - Track agent decisions, similarity scores, and API performance
   - Set up alerts for errors and performance issues
   <br>
    <div align="center">
        <a><img src="https://github.com/user-attachments/assets/e4224525-f380-4e45-b3d6-237c9abc8fa9"></a> 
    </div>
   </br>

## Features

### Document Processing
- **Multi-format support**: PDF, DOCX, PPTX, TXT, XLSX, CSV, SQL
- **Structured data handling**: CSV and XLSX files processed row by row
- **SQL file processing**: Extracts and indexes SQL queries and schemas
- **Unique chunk identification**: Prevents duplicate chunks in vector database

### Agentic RAG System
- **Intelligent routing**: Automatically chooses between RAG and web search
- **Similarity-based decisions**: Uses 70% threshold for source selection
- **Web search integration**: DuckDuckGo search for additional context
- **Graphics generation**: Creates charts and visualizations from data

### Observability
- **LangSmith tracing**: Complete LLM chain visibility and debugging
- **LogFire logging**: Structured logging with spans and metrics
- **Grafana dashboards**: Real-time system monitoring
- **Decision tracking**: User-visible source attribution

### User Experience
- **Transparent decisions**: Shows whether answers come from RAG or web
- **Graphics support**: Visual data representation
- **Real-time feedback**: Immediate response with source information
- **Session management**: Maintains context across interactions

### Technical Features
- **Vector similarity search**: Embedding-based retrieval
- **Scalable architecture**: Microservices design
- **Health monitoring**: Container health checks
- **Error handling**: Comprehensive error management
- **Performance optimization**: Async operations and caching
