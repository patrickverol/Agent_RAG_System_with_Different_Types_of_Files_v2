"""
Agent Module
This module implements the agentic RAG system with similarity-based routing.
It provides functions for RAG retrieval, web search, and answer generation.
Includes observability features with LogFire and LangSmith.
"""

# Import regular expression module
import re

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

# Import classes for LangGraph
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
import re

# LogFire and LangSmith for Observability
import logfire
from langsmith import traceable
from langsmith import Client as LangSmithClient

# Load environment variables
load_dotenv()

# Get API keys for observability
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
langchain_api_key_env = os.getenv("LANGCHAIN_API_KEY")

# Token LogFire
logfire_api_key = os.getenv("LOGFIRE_TOKEN")

# Get Google API key
google_api_key = os.getenv("GOOGLE_API_KEY")

# Configure LogFire
try:
    logfire.configure() 
    print("Agent Log - Logfire configured.") 
except Exception as e:
     print(f"Agent Log - Warning: Failed to configure Logfire automatically: {e}")

# Check environment variables for observability
if not langsmith_api_key or not langchain_api_key_env:
    print("Agent Log - Warning: LANGSMITH_API_KEY and/or LANGCHAIN_API_KEY not defined. LangSmith tracing may not work completely.")

if not logfire_api_key:
     print("Agent Log - Warning: LOGFIRE_API_KEY not defined. Logs to Pydantic LogFire Cloud will not work.")

# Check LANGCHAIN_TRACING_V2
if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() != "true":
    print("Agent Log - Warning: LANGCHAIN_TRACING_V2 environment variable is not set to 'true'. Automatic LangGraph tracing to LangSmith MAY be disabled.")

# Check LANGCHAIN_API_KEY specifically for tracing
if not langchain_api_key_env:
     print("Agent Log - Warning: LANGCHAIN_API_KEY environment variable not defined. LangGraph tracing to LangSmith will NOT work.")


# Define the state class for the graph (AI Agent)
class GraphState(TypedDict):
    query: str
    rag_context: list | None
    best_similarity_score: float | None
    web_results: str | None
    graphics_description: str | None
    graphics_result: str | None
    final_answer: str | None
    user_wants_graphics: bool | None
    source_decision: Literal["RAG", "WEB", ""]


@traceable(run_type="llm", name="Node_InitializeAgent")  # LangSmith Decorator
def initialize_agent_node(state: GraphState) -> dict:
    """
    Generic initialization node for the agent.
    This is a best practice for LangGraph initialization.
    """
    
    # Extract query from state
    query = state["query"]

    # Logfire span to group logs from this node
    span = logfire.span("Executing Node: Initialize Agent", query=query)
    
    # Within the Span
    with span:
        print("--- Node: Initialize Agent ---")
        print(f"  Initializing agent for query: {query}")
        
        logfire.info("Agent initialized", query=query, query_length=len(query))
        
        # Return empty dict to pass control to next node
        return {}


@traceable(run_type="retriever", name="Node_RetrieveRAG")  # LangSmith Decorator
def retrieve_rag_node(state: GraphState) -> dict:
    """
    Retrieves documents from RAG by calling the API endpoint.
    Returns the best context based on similarity score.
    """
    
    # Extract query from state
    query = state["query"]
    
    # Start Logfire span to track execution of this node
    span = logfire.span("Executing Node: Retrieve RAG", query=query)
    
    # Within the span context
    with span:
        print("--- Node: Retrieve RAG ---")
        
        try:
            # Call the RAG API
            url = "http://backend:8000/rag_api"
            payload = json.dumps({"query": query})
            headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
            
            response = requests.request("POST", url, headers=headers, data=payload)
            response.raise_for_status()
            
            response_data = json.loads(response.text)
            
            # Extract context from API response
            documents = response_data.get('context', [])
            
            if documents:
                # Get the best similarity score (first document has highest score)
                best_similarity_score = documents[0].get('similarity_score', 0)
                
                print(f"  RAG context found with {len(documents)} documents.")
                print(f"  Best similarity score: {best_similarity_score}%")
                
                logfire.info("RAG context found", 
                           documents_count=len(documents), 
                           best_similarity_score=best_similarity_score,
                           all_scores=[doc.get('similarity_score', 0) for doc in documents])
                
                return {
                    "rag_context": documents,
                    "best_similarity_score": best_similarity_score
                }
            else:
                print("  No RAG context found.")
                logfire.info("No RAG context found.")
                return {
                    "rag_context": [],
                    "best_similarity_score": 0.0
                }
                
        except Exception as e:
            logfire.error("Error in RAG node", query=query, error=str(e), exc_info=True)
            print(f"  Error in RAG node: {e}")
            return {
                "rag_context": [],
                "best_similarity_score": 0.0
            }


@traceable(run_type="tool", name="Node_SearchWeb")  # LangSmith Decorator
def search_web_node(state: GraphState) -> dict:
    """
    Searches the web for information.
    """
    
    # Extract query from state
    query = state["query"]
    
    # Start Logfire span to track execution of this node
    span = logfire.span("Executing Node: Search Web", query=query)
    
    # Within the span context
    with span:
        print("--- Node: Search Web ---")
        
        try:
            web_search_tool = DuckDuckGoSearchRun()
            results = web_search_tool.run(query)
            if not results:
                print("  No web search results.")
                logfire.info("No web search results.")
                return {"web_results": "No relevant results found on the web."}
            else:
                print(f"  Web search results found ({len(results)} chars).")
                logfire.info("Web search results found.", results_length=len(results))
                return {"web_results": results}
        except Exception as e:
            logfire.error("Error in Web Search node", query=query, error=str(e), exc_info=True)
            print(f"  Error in Web Search node: {e}")
            return {"web_results": f"Error performing web search: {e}"}


@traceable(run_type="llm", name="Node_GenerateGraphicsDescription")  # LangSmith Decorator
def generate_graphics_description_node(state: GraphState, groq_api_key: str) -> dict:
    """
    Generates graphics description using a specialized LLM.
    Can use RAG context or web results for data-driven visualizations.
    """
    
    # Extract query and contexts from state
    query = state["query"]
    rag_context = state.get("rag_context", [])
    web_results = state.get("web_results", "")
    
    # Start Logfire span to track execution of this node
    span = logfire.span("Executing Node: Generate Graphics Description", query=query)
    
    # Within the span context
    with span:
        print("--- Node: Generate Graphics Description ---")
        
        try:
            # Determine the best context to use for graphics generation
            context_to_use = ""
            context_source = "none"
            
            if rag_context and len(rag_context) > 0:
                # Use RAG context if available
                context_to_use = "\n\n".join([f"[{doc['id']}] {doc['content']}" for doc in rag_context])
                context_source = "rag"
                print("  Using RAG context for graphics generation...")
            elif web_results and web_results != "No relevant results found on the web.":
                # Use web results as fallback
                context_to_use = web_results
                context_source = "web"
                print("  Using web results for graphics generation...")
            else:
                print("  No context available for graphics generation...")
            
            if context_to_use:
                graphics_prompt = f"""You are a professional graphics designer and data visualization expert specializing in creating precise, data-driven charts.

                User Request: {query}

                Available Context:
                {context_to_use}

                CRITICAL INSTRUCTIONS:
                1. Use the EXACT numerical values extracted from the context above
                2. Determine the most appropriate chart type based on the data type and user request
                3. Create a SPECIFIC, DETAILED description for AI image generation with precise coordinates

                CHART TYPE RECOMMENDATIONS:
                - Stock data with dates: Line chart showing price over time
                - Financial metrics: Bar chart or line chart
                - Categorical data: Bar chart
                - Distribution data: Pie chart or histogram
                - Time series: Line chart with proper date formatting

                CHART SPECIFICATION REQUIREMENTS:
                Your description MUST include:
                1. EXACT chart type (line chart, bar chart, scatter plot, etc.)
                2. SPECIFIC numerical values for X and Y axes with proper scaling
                3. PRECISE data points with exact numbers from the extracted data
                4. Clear axis labels with units (USD, dates, percentages, etc.)
                5. Professional color scheme (blues for financial, grays for business)
                6. Grid lines and proper scaling
                7. Title and legend if applicable
                8. Data point positioning with exact coordinates

                FORMAT YOUR RESPONSE AS:
                Chart Type: [specific type]
                X-Axis: [exact values and labels with units]
                Y-Axis: [exact values and labels with units]
                Data Points: [exact numerical pairs with coordinates]
                Styling: [professional appearance details]
                Title: [descriptive title]
                Additional Notes: [any specific positioning or formatting requirements]

                Description:"""
                
                logfire.info("Generating context-based graphics description", 
                           context_source=context_source, 
                           context_length=len(context_to_use))
            else:
                # Create prompt for generic graphics description generation
                graphics_prompt = f"""You are a professional graphics designer and data visualization expert.
                
                User Request: {query}
                
                Based on the user's request, create a detailed description for AI image generation that will create a professional chart or visualization.
                
                Focus on:
                1. The type of chart/graph (line chart, bar chart, pie chart, etc.)
                2. Data visualization elements (axes, labels, data points, etc.)
                3. Professional styling (clean design, readable fonts, appropriate colors)
                4. Layout and composition
                5. Any specific data or values to be displayed
                
                Make your description clear and specific for AI image generation. Use terms that will help create a professional, business-ready visualization.
                
                Description:"""
                
                logfire.info("Generating generic graphics description", context_source="none")
            
            # Use a specialized LLM for description generation
            description_llm = ChatGroq(
                api_key=groq_api_key,
                model="llama3-8b-8192",
                temperature=0.3
            )
            
            # Generate the graphics description
            response = description_llm.invoke(graphics_prompt)
            graphics_description = response.content
            
            print(f"  Graphics description generated ({len(graphics_description)} chars).")
            logfire.info("Graphics description generated", 
                        description_length=len(graphics_description), 
                        context_source=context_source)
            
            return {"graphics_description": graphics_description}
        
        except Exception as e:
            logfire.error("Error in Graphics Description Generation node", query=query, error=str(e), exc_info=True)
            print(f"  Error in Graphics Description Generation node: {e}")
            return {"graphics_description": f"Error generating graphics description: {e}"}


@traceable(run_type="llm", name="Node_BuildGraphicsImage")  # LangSmith Decorator
def build_graphics_image_node(state: GraphState) -> dict:
    """
    Builds the actual image using Google GenAI image generation.
    """
    
    # Extract query and description from state
    query = state["query"]
    graphics_description = state.get("graphics_description")
    
    # Start Logfire span to track execution of this node
    span = logfire.span("Executing Node: Build Graphics Image", query=query)
    
    # Within the span context
    with span:
        print("--- Node: Build Graphics Image ---")
        
        try:
            # Import Google GenAI for image generation
            from google import genai
            from google.genai import types
            from PIL import Image
            from io import BytesIO
            import base64
            
            # Configure Google GenAI
            client = genai.Client()
            
            if not graphics_description or graphics_description.startswith("Error"):
                raise Exception("No valid graphics description available")
            
            print("  Generating image using Google GenAI...")
            
            # Create the image generation prompt
            image_prompt = f"""Create a professional business chart or data visualization based on this detailed specification: {graphics_description}
            
            User request: {query}
            
            CRITICAL REQUIREMENTS:
            - Use the EXACT numerical values provided in the description
            - Follow the specified chart type precisely
            - Include all data points with accurate positioning
            - Use the exact axis labels and units specified
            - Apply professional business styling with clean fonts
            - Include grid lines for better readability
            - Use appropriate color scheme (blues for financial data, grays for business)
            - Ensure proper scaling and spacing
            - Make it look like a professional business presentation chart
            
            IMPORTANT: The chart must be mathematically accurate with the exact data values provided. Do not approximate or round numbers unless specified."""
            
            # Generate the image using Google GenAI
            response = client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=image_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
            
            # Extract the generated image
            generated_image = None
            generated_text = ""
            
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    generated_text = part.text
                elif part.inline_data is not None:
                    # Convert the image data to base64 for storage and display
                    image = Image.open(BytesIO(part.inline_data.data))
                    
                    # Convert to base64
                    img_buffer = BytesIO()
                    image.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    img_str = base64.b64encode(img_buffer.getvalue()).decode()
                    
                    # Create image data URL
                    image_data_url = f"data:image/png;base64,{img_str}"
                    generated_image = image_data_url
            
            if generated_image:
                graphics_result = f"""GRAPHICS GENERATION RESULT:

                User Request: {query}

                Generated Description:
                {graphics_description}

                Generated Image:
                <image_data>{generated_image}</image_data>

                Generated Text: {generated_text}

                Status: Image generated successfully by Google GenAI"""
                
                print("  Image generated successfully using Google GenAI.")
                logfire.info("Image generated successfully", 
                           has_image=True, 
                           model="gemini-2.0-flash-preview-image-generation")
                
            else:
                raise Exception("No image was generated by Google GenAI")
            
            return {"graphics_result": graphics_result}
        
        except Exception as e:
            logfire.error("Error in Build Graphics Image node", query=query, error=str(e), exc_info=True)
            print(f"  Error in Build Graphics Image node: {e}")
            return {"graphics_result": f"Error building graphics image: {e}"}


@traceable(run_type="llm", name="Node_GenerateAnswer")  # LangSmith Decorator
def generate_answer_node(state: GraphState, groq_api_key: str) -> dict:
    """
    Generates the final answer for the user.
    """
    
    # Extract query from state
    query = state["query"]

    # Logfire span
    span = logfire.span("Executing Node: Answer Generation", query=query)
    
    # Within the span context
    with span:
        print("--- Node: Answer Generation ---") 
        
        rag_context = state.get("rag_context", [])
        web_results = state.get("web_results")
        context_provided = ""
        source_used = "None"

        # Determine if RAG or WEB contexts are useful
        rag_useful = rag_context and len(rag_context) > 0
        web_useful = web_results and web_results != "No relevant results found on the web."

        # Check condition and define attributes
        if rag_useful and not web_useful:
            # Build context from RAG documents
            context_parts = []
            for doc in rag_context:
                context_parts.append(f"[{doc['id']}] {doc['content']}")
            context_provided = f"Context from internal documents:\n{chr(10).join(context_parts)}"
            source_used = "RAG"
            print("  Using RAG context to generate answer.") 
            logfire.info("Using RAG context to generate answer.", 
                        documents_count=len(rag_context),
                        best_similarity_score=state.get("best_similarity_score", 0))

        elif web_useful:
            context_provided = f"Web search results:\n{web_results}"
            source_used = "WEB"
            print("  Using web results to generate answer.") 
            logfire.info("Using web results to generate answer.")

        else:
            context_provided = "No additional information found in available sources."
            source_used = ""
            print("  No useful context found to generate answer.") 
            logfire.info("No useful context found to generate answer.")

        prompt = f"""You are a technical support assistant. Answer the user's question clearly and concisely, using ONLY the information provided in the context below, if available.
        If the user asks for a chart, just ignore that.
        User Query: {query}
        {context_provided}
        Answer:"""

        try:
            # Create LLM instance for final answer
            llm_resposta_final = ChatGroq(
                api_key=groq_api_key, 
                model="meta-llama/llama-4-maverick-17b-128e-instruct", 
                temperature=0.1
            )
            response = llm_resposta_final.invoke(prompt)
            final_answer = response.content
            print(f"  Answer generated using source: {source_used}") 
            
            logfire.info("Final answer generated", 
                        source_used=source_used, 
                        answer_length=len(final_answer))

            return {"final_answer": final_answer, "source_decision": source_used}
        
        except Exception as e:
            logfire.error("Error in generation node", query=query, source_used=source_used, error=str(e), exc_info=True)
            print(f"  Error in generation node: {e}") 
            return {"final_answer": f"Sorry, an error occurred while generating the answer: {e}", "source_decision": "ERROR"}


def decide_after_rag_edge(state: GraphState) -> Literal["generate_answer_node", "search_web_node"]:
    """
    Decision node after RAG to determine if we should use RAG results or search web.
    """
    
    best_similarity_score = state.get("best_similarity_score", 0)
    
    logfire.debug("Conditional Edge: Deciding after RAG", best_similarity_score=best_similarity_score)
    print(f"--- Conditional Edge: After RAG, best similarity score = {best_similarity_score}% ---") 
    
    if best_similarity_score >= 85:
        print("  Edge: Similarity score >= 85%, using RAG context for answer generation.") 
        return "generate_answer_node"
    else:
        print("  Edge: Similarity score < 85%, searching web for better context.") 
        return "search_web_node"


def decide_graphics_edge(state: GraphState) -> Literal["generate_graphics_description_node", "end"]:
    """
    Decision node to determine if user wants graphics generation.
    """
    
    user_wants_graphics = state.get("user_wants_graphics", False)
    
    logfire.debug("Conditional Edge: Deciding graphics generation", user_wants_graphics=user_wants_graphics)
    print(f"--- Conditional Edge: User wants graphics = {user_wants_graphics} ---") 
    
    if user_wants_graphics:
        print("  Edge: User wants graphics, proceeding to graphics generation.") 
        return "generate_graphics_description_node"
    else:
        print("  Edge: User doesn't want graphics, ending process.") 
        return "end"


def compile_graph(groq_api_key: str):
    """
    Compiles (creates) the LangGraph (i.e., the AI Agent).
    """
    
    # Logfire span
    span = logfire.span("Compiling LangGraph")
    
    # Within the span context
    with span:
        print("Compiling LangGraph...") 
        
        try:
            # Create nodes
            graph_builder = StateGraph(GraphState)
            graph_builder.add_node("initialize_agent_node", initialize_agent_node)
            graph_builder.add_node("retrieve_rag_node", retrieve_rag_node)
            graph_builder.add_node("search_web_node", search_web_node)
            graph_builder.add_node("generate_graphics_description_node", lambda state: generate_graphics_description_node(state, groq_api_key))
            graph_builder.add_node("build_graphics_image_node", build_graphics_image_node)
            graph_builder.add_node("generate_answer_node", lambda state: generate_answer_node(state, groq_api_key))
            graph_builder.set_entry_point("initialize_agent_node")
            
            # Add edges
            graph_builder.add_edge("initialize_agent_node", "retrieve_rag_node")
            graph_builder.add_edge("search_web_node", "generate_answer_node")
            graph_builder.add_edge("generate_graphics_description_node", "build_graphics_image_node")
            graph_builder.add_edge("build_graphics_image_node", END)
            
            # Add conditional edge after RAG
            graph_builder.add_conditional_edges("retrieve_rag_node", decide_after_rag_edge, {
                "generate_answer_node": "generate_answer_node",
                "search_web_node": "search_web_node",
            })
            
            # Add conditional edge after answer generation for graphics
            graph_builder.add_conditional_edges("generate_answer_node", decide_graphics_edge, {
                "generate_graphics_description_node": "generate_graphics_description_node",
                "end": END,
            })
            
            # Compile graph
            app = graph_builder.compile()
            print("Graph compiled successfully!") 
            
            logfire.info("LangGraph compiled successfully")
            return app
        except Exception as e:
            print(f"Agent Log - CRITICAL error compiling graph: {e}")  # Print for critical error
            logfire.critical("Error compiling LangGraph", error=str(e), exc_info=True)
            raise Exception(f"Error compiling graph: {e}")


def run_agent(query: str, groq_api_key: str, user_wants_graphics: bool = False):
    """
    Runs the agent with the given query and returns the final state.
    
    Args:
        query (str): The user query
        groq_api_key (str): Groq API key for LLM operations
        user_wants_graphics (bool): Whether user wants graphics generation
    """
    # Create Logfire span to track the entire agent execution
    span = logfire.span("Processing user query with Agent", query=query)
    
    # Within the span context
    with span:
        try:
            logfire.info("Starting agent execution", query=query, user_wants_graphics=user_wants_graphics)
            
            # Compile the graph
            app = compile_graph(groq_api_key)
            
            # Prepare inputs
            inputs = {
                "query": query,
                "user_wants_graphics": user_wants_graphics
            }

            # Execute the graph (AI Agent)
            final_state = app.invoke(inputs)
            
            logfire.info("Agent execution completed successfully", 
                        query=query, 
                        source_decision=final_state.get("source_decision", "UNKNOWN"),
                        has_final_answer=bool(final_state.get("final_answer")),
                        has_graphics_result=bool(final_state.get("graphics_result")))
            
            return final_state
            
        except Exception as e:
            logfire.error("Error running agent", query=query, error=str(e), exc_info=True)
            raise Exception(f"Error running agent: {e}") 