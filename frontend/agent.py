"""
Agent Module
This module implements the agentic RAG system with LLM routing.
It provides functions for query routing, RAG retrieval, web search, and answer generation.
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
    source_decision: Literal["RAG", "WEB", "GRAPHICS", "RAG_GRAPHICS", ""]
    rag_context: str | None
    web_results: str | None
    graphics_description: str | None
    graphics_result: str | None
    final_answer: str | None


@traceable(run_type="llm", name="Node_RouteQuery")  # LangSmith Decorator
def dsa_route_query_node(state: GraphState, groq_api_key: str) -> dict:
    """
    Analyzes the query and decides the data source (RAG or WEB).
    Updates 'source_decision' in the state.
    """
    
    # Extract query from state
    query = state["query"]

    # Logfire span to group logs from this node
    span = logfire.span("Executing Node: Query Routing", query=query)
    
    # Within the Span
    with span:
        print("--- Node: Query Routing ---")

        # **REFINED PROMPT WITH EXAMPLES (FEW-SHOT)**
        prompt = f"""Your task is to classify a user query to direct it to the best information source. The sources are:
        1.  **RAG**: Internal knowledge base with technical support documents, specific procedures, system configurations, internal guides. Use RAG for questions about 'how to do X in our system', 'what is the configuration of Y', 'internal documentation about Z'.
        2.  **WEB**: General web search for information about third-party software (e.g., Anaconda, Python, Excel), technology news, generic errors not documented internally, very recent information, or anything that is not specific to our internal documents.
        3.  **GRAPHICS**: Requests to create, generate, build, draw, visualize, chart, graph, diagram, or any visual representation that doesn't need specific data. Use GRAPHICS for generic requests like 'create a chart template', 'draw a diagram of a computer', 'make a generic visualization'.
        4.  **RAG_GRAPHICS**: Requests to create visualizations using specific data from our internal knowledge base. Use RAG_GRAPHICS when the user asks to visualize specific data, create charts from our data, or generate graphics based on internal information like 'create a chart of our stock data', 'visualize the financial metrics', 'generate a graph from our database'.

        Examples:
        - Query: "What is a broker?" -> Answer: RAG
        - Query: "What is the latest informations about (something)?" -> Answer: WEB
        - Query: "Can you provide me financial or marketmetrics about (something)?" -> Answer: RAG
        - Query: "Could you provide me with the closing prices of (something) shares?" -> Answer: RAG
        - Query: "How do I (something)?" -> Answer: WEB
        - Query: "How to (something)?" -> Answer: WEB
        - Query: "Could you provide me with the closing prices of (something) shares? Then, create a line graph for me with these values." -> Answer: RAG_GRAPHICS
        - Query: "Please, find some document about (something), and then create a chart with this data." -> Answer: RAG_GRAPHICS
        - Query: "Create a generic chart template" -> Answer: GRAPHICS
        - Query: "Draw a diagram of a computer" -> Answer: GRAPHICS

        Now, classify the following query:
        User Query: '{query}'

        Based on the query, what is the most appropriate source? Answer ONLY with the word 'RAG', 'WEB', 'GRAPHICS', or 'RAG_GRAPHICS'."""

        try:
            # **CREATE DEDICATED LLM FOR ROUTING WITH HIGHER TEMPERATURE**
            # Here we can use a simpler model, like an SLM
            router_llm = ChatGroq(api_key=groq_api_key,
                                  model="llama3-8b-8192",
                                  temperature=0.4)

            # Execute the router
            response = router_llm.invoke(prompt)

            # Extract the response
            raw_decision = response.content 

            # DEBUG PRINT ESSENTIAL (Check in console!)
            print(f"DEBUG: LLM Decision (Request Router): '{raw_decision}'")

            # Clean the response text to keep only the word we're interested in
            decision = raw_decision.strip().upper().replace("'", "").replace('"', '') 

            # Final decision logic (if not RAG, assume WEB)
            if decision == "RAG":
                final_decision = "RAG"
            elif decision == "WEB":
                final_decision = "WEB"
            elif decision == "GRAPHICS":
                final_decision = "GRAPHICS"
            elif decision == "RAG_GRAPHICS":
                final_decision = "RAG_GRAPHICS"
            else:
                # If not RAG, WEB, GRAPHICS, or RAG_GRAPHICS, log the value but go to WEB
                logfire.warn("Unexpected decision from router, using WEB as fallback.", raw_decision=raw_decision, query=query, decision_parsed=decision)
                print(f"  Invalid/unexpected decision from router: '{raw_decision}'. Using WEB as fallback.") 
                final_decision = "WEB"

            print(f"  Final Router Decision: {final_decision}") 
            
            logfire.info("Routing decision finalized", raw_decision=raw_decision, final_decision=final_decision)

            return {"source_decision": final_decision}

        except Exception as e:
            logfire.error("Error in routing node, using WEB as fallback.", query=query, error=str(e), exc_info=True)
            print(f"  Error in routing node: {e}") 
            print("  Using WEB as fallback due to error.") 
            return {"source_decision": "WEB"}


@traceable(run_type="retriever", name="Node_RetrieveRAG")  # LangSmith Decorator
def dsa_retrieve_rag_node(state: GraphState) -> dict:
    """
    Retrieves documents from RAG by calling the API endpoint.
    """
    
    # Extract query from state
    query = state["query"]
    
    # Start Logfire span to track execution of this node
    span = logfire.span("Executing Node: Retrieve RAG", query=query)
    
    # Within the span context
    with span:
        try:
            # Call the RAG API instead of using local RAG
            url = "http://backend:8000/rag_api"
            payload = json.dumps({"query": query})
            headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
            
            response = requests.request("POST", url, headers=headers, data=payload)
            response.raise_for_status()
            
            response_data = json.loads(response.text)
            
            # Extract context from API response
            documents = response_data.get('context', [])
            answer = response_data.get('answer', '')
            
            if documents:
                # Build context from documents
                context = "\n\n".join([f"[{doc['id']}] {doc['content']}" for doc in documents])
                print(f"  RAG context found ({len(context)} chars).") 
                logfire.info("RAG context found.", context_length=len(context))
                return {"rag_context": context}
            else:
                print("  No RAG context found.") 
                logfire.info("No RAG context found.")
                return {"rag_context": "No relevant internal documents were found."}
                
        except Exception as e:
            logfire.error("Error in RAG node", query=query, error=str(e), exc_info=True)
            print(f"  Error in RAG node: {e}") 
            return {"rag_context": f"Error searching internal documents: {e}"}


@traceable(run_type="tool", name="Node_SearchWeb")  # LangSmith Decorator
def dsa_search_web_node(state: GraphState) -> dict:
    """
    Searches the web for information.
    """
    
    # Extract query from state
    query = state["query"]
    
    # Start Logfire span to track execution of this node
    span = logfire.span("Executing Node: Search Web", query=query)
    
    # Within the span context
    with span:
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
def dsa_generate_graphics_description_node(state: GraphState, groq_api_key: str) -> dict:
    """
    Generates graphics description using a specialized LLM.
    Can use RAG context if available for data-driven visualizations.
    """
    
    # Extract query and RAG context from state
    query = state["query"]
    rag_context = state.get("rag_context", "")
    source_decision = state.get("source_decision", "")
    
    # Start Logfire span to track execution of this node
    span = logfire.span("Executing Node: Generate Graphics Description", query=query)
    
    # Within the span context
    with span:
        print("--- Node: Generate Graphics Description ---")
        
        try:
            # Check if we have RAG context and if this is a RAG_GRAPHICS request
            has_rag_data = rag_context and rag_context != "No relevant internal documents were found." and not rag_context.startswith("Error")
            is_rag_graphics = source_decision == "RAG_GRAPHICS"
            
            if has_rag_data and is_rag_graphics:
                print("  Using RAG data for data-driven graphics generation...")
                
                graphics_prompt = f"""You are a professional graphics designer and data visualization expert specializing in creating precise, data-driven charts.

                User Request: {query}

                Available Data from Internal Knowledge Base:
                {rag_context}

                CRITICAL INSTRUCTIONS:
                1. Use the EXACT numerical values extracted above
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
                
                logfire.info("Generating RAG-based graphics description", has_rag_data=True, source_decision=source_decision)
            else:
                print("  Generating generic graphics description...")
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
                
                logfire.info("Generating generic graphics description", has_rag_data=False, source_decision=source_decision)
            
            # Use a specialized LLM for description generation
            description_llm = ChatGroq(
                api_key=groq_api_key,
                model="llama3-8b-8192",  # Using a more capable model for complex tasks
                temperature=0.3
            )
            
            # Generate the graphics description
            response = description_llm.invoke(graphics_prompt)
            graphics_description = response.content
            
            print(f"  Graphics description generated ({len(graphics_description)} chars).")
            logfire.info("Graphics description generated", description_length=len(graphics_description), used_rag_data=has_rag_data and is_rag_graphics)
            
            return {"graphics_description": graphics_description}
        
        except Exception as e:
            logfire.error("Error in Graphics Description Generation node", query=query, error=str(e), exc_info=True)
            print(f"  Error in Graphics Description Generation node: {e}")
            return {"graphics_description": f"Error generating graphics description: {e}"}


@traceable(run_type="llm", name="Node_BuildGraphicsImage")  # LangSmith Decorator
def dsa_build_graphics_image_node(state: GraphState, google_api_key: str) -> dict:
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
                logfire.info("Image generated successfully", has_image=True, model="gemini-2.0-flash-preview-image-generation")
                
            else:
                raise Exception("No image was generated by Google GenAI")
            
            return {"graphics_result": graphics_result}
        
        except Exception as e:
            logfire.error("Error in Build Graphics Image node", query=query, error=str(e), exc_info=True)
            print(f"  Error in Build Graphics Image node: {e}")
            return {"graphics_result": f"Error building graphics image: {e}"}


@traceable(run_type="llm", name="Node_GenerateAnswer")  # LangSmith Decorator
def dsa_generate_answer_node(state: GraphState, groq_api_key: str) -> dict:
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
        
        rag_context = state.get("rag_context")
        web_results = state.get("web_results")
        graphics_result = state.get("graphics_result")
        context_provided = ""
        source_used = "None"

        # Determine if RAG, WEB, and GRAPHICS contexts are useful
        rag_useful = rag_context and rag_context != "No relevant internal documents were found."
        web_useful = web_results and web_results != "No relevant results found on the web."
        graphics_useful = graphics_result and not graphics_result.startswith("Error")

        # Check condition and define attributes
        if graphics_useful and rag_useful:
            context_provided = f"Context from internal documents:\n{rag_context}"
            source_used = "RAG_GRAPHICS"
            print("  Using RAG context to generate answer.") 
            logfire.info("Using RAG context to generate answer.")
            
        elif graphics_useful and not rag_useful:
            source_used = "GRAPHICS"
            print("  Generating answer without RAG context.") 
            logfire.info("Generating answer without RAG context.")
        
        elif rag_useful:
            context_provided = f"Context from internal documents:\n{rag_context}"
            source_used = "RAG"
            print("  Using RAG context to generate answer.") 
            logfire.info("Using RAG context to generate answer.")

        elif web_useful:
            context_provided = f"Web search results:\n{web_results}"
            source_used = "WEB"
            print("  Using web results to generate answer.") 
            logfire.info("Using web results to generate answer.")

        else:
            context_provided = "No additional information found in available sources."
            print("  No useful context found to generate answer.") 
            logfire.info("No useful context found to generate answer.")

        # Log the source used
        logfire.info('Source(s) for generation', source_used=source_used, rag_context_present=rag_useful, web_results_present=web_useful, graphics_present=graphics_useful)

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
            
            logfire.info("Final answer generated", source_used=source_used, answer_length=len(final_answer))

            if graphics_useful:
                # For graphics, we need to preserve the image data in the final answer
                # Extract the image data from graphics_result
                import re
                image_pattern = r'<image_data>(.*?)</image_data>'
                image_match = re.search(image_pattern, graphics_result, re.DOTALL)
                
                if image_match and rag_useful:
                    image_data = image_match.group(1)
                    # Create a final answer that includes the image data
                    final_answer = f"""{response.content}
                    <image_data>{image_data}</image_data>"""
    
                    return {"final_answer": final_answer}
                
                elif image_match and not rag_useful:
                    image_data = image_match.group(1)
                    # Create a final answer that includes the image data
                    final_answer = f"""{response.content}
                    <image_data>{image_data}</image_data>"""
                    return {"final_answer": final_answer}
                else:
                    # Fallback if no image data found
                    final_answer = f"I've processed your graphics request: '{query}'. However, there was an issue generating the image. Please try again."
                    return {"final_answer": final_answer}

            else:
                return {"final_answer": final_answer}
        
        except Exception as e:
            logfire.error("Error in generation node", query=query, source_used=source_used, error=str(e), exc_info=True)
            print(f"  Error in generation node: {e}") 
            return {"final_answer": f"Sorry, an error occurred while generating the answer: {e}"}


def dsa_decide_source_edge(state: GraphState) -> Literal["retrieve_rag_node", "search_web_node", "generate_graphics_description_node"]:
    """
    Decision node for source (RAG, WEB, GRAPHICS, or RAG_GRAPHICS).
    """
    
    decision = state["source_decision"]
    
    logfire.debug("Conditional Edge: Deciding next node", current_decision=decision)  # LogFire debug level
    print(f"--- Conditional Edge: Decision received = '{decision}' ---") 
    
    if decision == "RAG":
        print("  Edge: Going to RAG.") 
        return "retrieve_rag_node"
    elif decision == "GRAPHICS":
        print("  Edge: Going to Graphics Description Generation.") 
        return "generate_graphics_description_node"
    elif decision == "RAG_GRAPHICS":
        print("  Edge: Going to RAG first, then Graphics.") 
        return "retrieve_rag_node"
    else:
        print("  Edge: Going to WEB.") 
        return "search_web_node"


def dsa_decide_after_rag_edge(state: GraphState) -> Literal["generate_answer_node", "generate_graphics_description_node"]:
    """
    Decision node after RAG to determine if we need graphics generation.
    """
    
    decision = state["source_decision"]
    
    logfire.debug("Conditional Edge: Deciding after RAG", current_decision=decision)
    print(f"--- Conditional Edge: After RAG, decision was = '{decision}' ---") 
    
    if decision == "RAG_GRAPHICS":
        print("  Edge: Going to Graphics Description Generation with RAG data.") 
        return "generate_graphics_description_node"
    else:
        print("  Edge: Going to Answer Generation.") 
        return "generate_answer_node"


def dsa_compile_graph(groq_api_key: str):
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
            graph_builder.add_node("route_query_node", lambda state: dsa_route_query_node(state, groq_api_key))
            graph_builder.add_node("retrieve_rag_node", dsa_retrieve_rag_node)
            graph_builder.add_node("search_web_node", dsa_search_web_node)
            graph_builder.add_node("generate_graphics_description_node", lambda state: dsa_generate_graphics_description_node(state, groq_api_key))
            graph_builder.add_node("build_graphics_image_node", lambda state: dsa_build_graphics_image_node(state, google_api_key))
            graph_builder.add_node("generate_answer_node", lambda state: dsa_generate_answer_node(state, groq_api_key))
            graph_builder.set_entry_point("route_query_node")
            
            # Create condition for routing
            graph_builder.add_conditional_edges("route_query_node", dsa_decide_source_edge, {
                "retrieve_rag_node": "retrieve_rag_node",
                "search_web_node": "search_web_node",
                "generate_graphics_description_node": "generate_graphics_description_node",
            })
            
            # Add conditional edge after RAG to handle RAG_GRAPHICS
            graph_builder.add_conditional_edges("retrieve_rag_node", dsa_decide_after_rag_edge, {
                "generate_answer_node": "generate_answer_node",
                "generate_graphics_description_node": "generate_graphics_description_node",
            })
            
            # Add edges
            graph_builder.add_edge("search_web_node", "generate_answer_node")
            graph_builder.add_edge("generate_graphics_description_node", "build_graphics_image_node")
            graph_builder.add_edge("build_graphics_image_node", "generate_answer_node")
            graph_builder.add_edge("generate_answer_node", END)
            
            # Compile graph
            app = graph_builder.compile()
            print("Graph compiled successfully!") 
            
            logfire.info("LangGraph compiled successfully")
            return app
        except Exception as e:
            print(f"Agent Log - CRITICAL error compiling graph: {e}")  # Print for critical error
            logfire.critical("Error compiling LangGraph", error=str(e), exc_info=True)
            raise Exception(f"Error compiling graph: {e}")


def run_agent(query: str, groq_api_key: str):
    """
    Runs the agent with the given query and returns the final state.
    """
    # Create Logfire span to track the entire agent execution
    span = logfire.span("Processing user query with Agent", query=query)
    
    # Within the span context
    with span:
        try:
            logfire.info("Starting agent execution", query=query)
            
            # Compile the graph
            app = dsa_compile_graph(groq_api_key)
            
            # Prepare inputs
            inputs = {"query": query}

            # Execute the graph (AI Agent)
            final_state = app.invoke(inputs)
            
            logfire.info("Agent execution completed successfully", 
                        query=query, 
                        source_decision=final_state.get("source_decision"),
                        has_final_answer=bool(final_state.get("final_answer")))
            
            return final_state
            
        except Exception as e:
            logfire.error("Error running agent", query=query, error=str(e), exc_info=True)
            raise Exception(f"Error running agent: {e}") 