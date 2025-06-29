"""
Web Application
This module implements the Streamlit web interface for the RAG system.
It provides a user-friendly interface for querying documents and viewing results.
"""

# Import regular expression module
import re

# Import Streamlit for web interface
import streamlit as st

# Import HTTP request handling
import requests

# Import JSON handling
import json

# Import time for performance measurement
import time

# Import environment variables handling
import os

# Import function to load environment variables
from dotenv import load_dotenv

# Filter warnings
import warnings
warnings.filterwarnings('ignore')

# Import evaluation and feedback modules
from llm import gera_documento_id, captura_user_input, captura_user_feedback

# Import storage module
from storage import get_storage

# Import agent module
from agent import run_agent

# Load environment variables
load_dotenv()


def main():
    """
    Main function to run the Streamlit application.
    Handles user interface, document queries, and feedback collection.
    """
    
    # Configure Streamlit page
    st.set_page_config(page_title="RAG Project", page_icon=":100:", layout="wide")

    # Set application title
    st.title('🤖 RAG Project')
    st.title('🔍 Search Engine with Generative AI and RAG')

    # Get Groq API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key.strip() == "":
        st.error("GROQ_API_KEY not defined or empty in environment variables!")
        st.error("Please set the GROQ_API_KEY environment variable in your docker-compose.yaml or .env file.")
        st.stop()

    # Sidebar with instructions
    st.sidebar.title("Instructions")
    st.sidebar.markdown("""
    ### How to Use the App:
    - Type a question in the text input field.
    - Click on the "Send" button.
    - View the answer and the documents used to answer the question.
    - Click on the "Satisfied" button if you are satisfied with the answer.
    - Click on the "Not Satisfied" button if you are not satisfied with the answer.
    - Click on the "Download" button to download the document.

    ### Available files:
    - PDF
    - DOCX
    - DOC
    - TXT
    - PPTX
    - PPT
    - CSV
    - XLSX

    ### Purpose:
    This application provides a search engine for documents. You can search for documents by typing a question in the text input field. Download the document to view the reference content.
    """)

    # Support button in sidebar
    if st.sidebar.button("Support"):
        st.sidebar.write("For any questions, please contact: patrickverol@gmail.com")

    # Configure storage
    storage_config = {
        'storage_type': 'http',
        'base_url': 'http://document_storage:8080'
    }
    
    # Add S3 specific configurations if needed
    if storage_config['storage_type'] == 's3':
        storage_config.update({
            'bucket_name': os.getenv('S3_BUCKET_NAME'),
            'region_name': os.getenv('AWS_REGION'),
            'endpoint_url': os.getenv('S3_ENDPOINT_URL')
        })
    # Add HTTP base URL if needed
    elif storage_config['storage_type'] == 'http':
        print(f"Using document storage URL: {storage_config['base_url']}")  # Debug log

    # Initialize storage
    storage = get_storage(**storage_config)
    
    # Only print base_url for HTTP storage
    if storage_config['storage_type'] == 'http':
        print(f"Using document storage URL: {storage.base_url}")

    # Initialize session state variables
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'docId' not in st.session_state:
        st.session_state.docId = None
    if 'userInput' not in st.session_state:
        st.session_state.userInput = ""
    if 'feedbackSubmitted' not in st.session_state:
        st.session_state.feedbackSubmitted = False

    # Create text input for questions with a default example
    question = st.text_input(
        "Type a question to execute a query on the documents:",
        value="What is a broker?",
        help="Example questions: 'What is a stock?', 'What are the key investment strategies?'"
    )

    # Check if "Send" button was clicked
    if st.button("Send"):
        if not question:
            st.warning("Type your question to continue.")
            return
            
        # Display the question
        st.write("The question was: \"", question+"\"")
        
        # Start time measurement
        start_time = time.time()
        
        try:
            with st.spinner("Processing your query..."):
                # Run the agent
                final_state = run_agent(question, groq_api_key)
                
                # Get the final answer
                answer = final_state.get("final_answer", "No answer generated.")
                source_decision = final_state.get("source_decision", "Unknown")
                
                end_time = time.time()
                responseTime = round(end_time - start_time, 2)
                
                # Display answer using markdown
                st.markdown(answer)
                
                # Display source information
                st.info(f"Source used (decided by router): {source_decision}")
                
                # If RAG was used, try to get documents for display
                if source_decision == "RAG":
                    try:
                        # Call the RAG API to get document details for display
                        url = "http://backend:8000/rag_api"
                        payload = json.dumps({"query": question})
                        headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
                        
                        response = requests.request("POST", url, headers=headers, data=payload)
                        response.raise_for_status()
                        
                        response_data = json.loads(response.text)
                        documents = response_data.get('context', [])
                        
                        if documents:
                            st.subheader("Documents used:")
                            
                            # Display expanded documents with download buttons
                            for doc in documents:
                                # Create expander for each document
                                with st.expander(f"{doc['id']} - {doc['path']}"):
                                    # Display document content
                                    st.write(doc['content'])
                                    
                                    # Get document URL from storage
                                    try:
                                        # Ensure the document path is relative and properly formatted
                                        doc_path = doc['path'].lstrip('/')
                                        print(f"Getting URL for document path: {doc_path}")
                                        doc_url = storage.get_document_url(doc_path)
                                        print(f"Generated URL: {doc_url}")
                                        
                                        # Get document content
                                        temp_file = storage.get_document(doc_path)
                                        print(f"Successfully retrieved document content")
                                        
                                        # Read the file content in binary mode
                                        with open(temp_file, 'rb') as f:
                                            doc_content = f.read()
                                        
                                        # Determine MIME type based on file extension
                                        file_ext = os.path.splitext(doc_path)[1].lower()
                                        mime_type = {
                                            '.pdf': 'application/pdf',
                                            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                                            '.doc': 'application/msword',
                                            '.txt': 'text/plain',
                                            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                                            '.ppt': 'application/vnd.ms-powerpoint'
                                        }.get(file_ext, 'application/octet-stream')
                                        
                                        # Create download button with proper MIME type
                                        st.download_button(
                                            label=f"Download {os.path.basename(doc_path)}",
                                            data=doc_content,
                                            file_name=os.path.basename(doc_path),
                                            mime=mime_type
                                        )
                                        
                                        # Clean up temporary file
                                        os.unlink(temp_file)
                                        
                                    except Exception as e:
                                        print(f"Error downloading {doc_path}: {str(e)}")
                                        st.error(f"Error downloading document: {str(e)}")
                    
                    except Exception as e:
                        print(f"Error getting document details: {e}")
                        st.warning("Could not retrieve document details for display.")

                # Add evaluation and feedback
                try:
                    # Generate document ID
                    docId = gera_documento_id(question, answer)
                    
                    # Capture user input
                    captura_user_input(
                        docId,
                        question.replace("'", ""), 
                        answer, 
                        1.0,  # Default score
                        responseTime,  # Use calculated response time
                    )

                    # Update session state
                    st.session_state.result = answer
                    st.session_state.docId = docId
                    st.session_state.userInput = question.replace("'", "")
                    st.session_state.feedbackSubmitted = False

                except Exception as e:
                    print(e)
                    st.error("Error processing the evaluation. Check the Qdrant and try again.")

        except Exception as e:
            st.error(f"Error processing your query: {str(e)}")

    # Display query result and feedback outside the "Send" button block
    if st.session_state.result:
        # Satisfaction feedback section
        if not st.session_state.feedbackSubmitted:
            st.write("Are you satisfied with the answer?")
            feedback_col1, feedback_col2 = st.columns(2)
            with feedback_col1:
                if st.button("Satisfied"):
                    captura_user_feedback(st.session_state.docId, st.session_state.userInput, st.session_state.result, True)
                    st.session_state.feedbackSubmitted = True
                    st.success("Feedback registered: Satisfied")
            with feedback_col2:
                if st.button("Not Satisfied"):
                    captura_user_feedback(st.session_state.docId, st.session_state.userInput, st.session_state.result, False)
                    st.session_state.feedbackSubmitted = True
                    st.warning("Feedback registered: Not Satisfied")


if __name__ == "__main__":
    main()