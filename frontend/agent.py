"""
Agent Module
This module implements the agentic RAG system with LLM routing.
It provides functions for query routing, RAG retrieval, web search, and answer generation.
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

# Filter warnings
import warnings
warnings.filterwarnings('ignore')

# Import classes for LangGraph
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal


# Define the state class for the graph (AI Agent)
class GraphState(TypedDict):
    query: str
    source_decision: Literal["RAG", "WEB", ""]
    rag_context: str | None
    web_results: str | None
    final_answer: str | None


def dsa_route_query_node(state: GraphState, groq_api_key: str) -> dict:
    """
    Analyzes the query and decides the data source (RAG or WEB).
    Updates 'source_decision' in the state.
    """
    
    print("--- Node: Query Routing ---")
    
    query = state["query"]

    # **REFINED PROMPT WITH EXAMPLES (FEW-SHOT)**
    prompt = f"""Your task is to classify a user query to direct it to the best information source. The sources are:
    1.  **RAG**: Internal knowledge base with technical support documents, specific procedures, system configurations, internal guides. Use RAG for questions about 'how to do X in our system', 'what is the configuration of Y', 'internal documentation about Z'.
    2.  **WEB**: General web search for information about third-party software (e.g., Anaconda, Python, Excel), technology news, generic errors not documented internally, very recent information, or anything that is not specific to our internal documents.

    Examples:
    - Query: "What is a broker?" -> Answer: RAG
    - Query: "What is the latest informations about (something)?" -> Answer: WEB
    - Query: "Can you provide me financial or marketmetrics about (something)?" -> Answer: RAG
    - Query: "How do I (something)?" -> Answer: WEB
    - Query: "How to (something)?" -> Answer: WEB

    Now, classify the following query:
    User Query: '{query}'

    Based on the query, what is the most appropriate source? Answer ONLY with the word 'RAG' or the word 'WEB'."""

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
        else:
            # If not RAG, and also not WEB, log the value but go to WEB
            if decision != "WEB":
                 print(f"  Invalid/unexpected decision from router: '{raw_decision}'. Using WEB as fallback.") 
            final_decision = "WEB"

        print(f"  Final Router Decision: {final_decision}") 

        return {"source_decision": final_decision}

    except Exception as e:
        print(f"  Error in routing node: {e}") 
        print("  Using WEB as fallback due to error.") 
        return {"source_decision": "WEB"}


def dsa_retrieve_rag_node(state: GraphState) -> dict:
    """
    Retrieves documents from RAG by calling the API endpoint.
    """
    
    query = state["query"]
    
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
            return {"rag_context": context}
        else:
            print("  No RAG context found.") 
            return {"rag_context": "No relevant internal documents were found."}
            
    except Exception as e:
        print(f"  Error in RAG node: {e}") 
        return {"rag_context": f"Error searching internal documents: {e}"}


def dsa_search_web_node(state: GraphState) -> dict:
    """
    Searches the web for information.
    """
    
    query = state["query"]
    
    try:
        web_search_tool = DuckDuckGoSearchRun()
        results = web_search_tool.run(query)
        if not results:
            print("  No web search results.") 
            return {"web_results": "No relevant results found on the web."}
        else:
            print(f"  Web search results found ({len(results)} chars).") 
            return {"web_results": results}
    except Exception as e:
        print(f"  Error in Web Search node: {e}") 
        return {"web_results": f"Error performing web search: {e}"}


def dsa_generate_answer_node(state: GraphState, groq_api_key: str) -> dict:
    """
    Generates the final answer for the user.
    """
    
    print("--- Node: Answer Generation ---") 
    
    query = state["query"]
    rag_context = state.get("rag_context")
    web_results = state.get("web_results")
    context_provided = ""
    source_used = "None"

    if rag_context and rag_context != "No relevant internal documents were found.":
        context_provided = f"Context from internal documents:\n{rag_context}"
        source_used = "RAG"
        print("  Using RAG context to generate answer.") 
    elif web_results and web_results != "No relevant results found on the web.":
        context_provided = f"Web search results:\n{web_results}"
        source_used = "WEB"
        print("  Using web results to generate answer.") 
    else:
        context_provided = "No additional information found in available sources."
        print("  No useful context found to generate answer.") 

    prompt = f"""You are a technical support assistant. Answer the user's question clearly and concisely, using ONLY the information provided in the context below, if available.
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
        return {"final_answer": final_answer}
    except Exception as e:
        print(f"  Error in generation node: {e}") 
        return {"final_answer": f"Sorry, an error occurred while generating the answer: {e}"}


def dsa_decide_source_edge(state: GraphState) -> Literal["retrieve_rag_node", "search_web_node"]:
    """
    Decision node for source (RAG or WEB).
    """
    
    decision = state["source_decision"]
    
    print(f"--- Conditional Edge: Decision received = '{decision}' ---") 
    
    if decision == "RAG":
        print("  Edge: Going to RAG.") 
        return "retrieve_rag_node"
    else:
        print("  Edge: Going to WEB.") 
        return "search_web_node"


def dsa_compile_graph(groq_api_key: str):
    """
    Compiles (creates) the LangGraph (i.e., the AI Agent).
    """
    
    print("Compiling LangGraph...") 
    
    # Create nodes
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("route_query_node", lambda state: dsa_route_query_node(state, groq_api_key))
    graph_builder.add_node("retrieve_rag_node", dsa_retrieve_rag_node)
    graph_builder.add_node("search_web_node", dsa_search_web_node)
    graph_builder.add_node("generate_answer_node", lambda state: dsa_generate_answer_node(state, groq_api_key))
    graph_builder.set_entry_point("route_query_node")
    
    # Create condition for routing
    graph_builder.add_conditional_edges("route_query_node", dsa_decide_source_edge, {
        "retrieve_rag_node": "retrieve_rag_node",
        "search_web_node": "search_web_node",
    })
    
    # Add edges
    graph_builder.add_edge("retrieve_rag_node", "generate_answer_node")
    graph_builder.add_edge("search_web_node", "generate_answer_node")
    graph_builder.add_edge("generate_answer_node", END)
    
    # Compile graph
    try:
        app = graph_builder.compile()
        print("Graph compiled successfully!") 
        return app
    except Exception as e:
        raise Exception(f"Error compiling graph: {e}")


def run_agent(query: str, groq_api_key: str):
    """
    Runs the agent with the given query and returns the final state.
    """
    try:
        # Compile the graph
        app = dsa_compile_graph(groq_api_key)
        
        # Prepare inputs
        inputs = {"query": query}

        # Execute the graph (AI Agent)
        final_state = app.invoke(inputs)
        
        return final_state
        
    except Exception as e:
        raise Exception(f"Error running agent: {e}") 