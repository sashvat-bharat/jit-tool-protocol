import os
import json
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Initialize the official Gemini Client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# 1. The Tool Registry (Your Infinite Shelf)
# for testing we added 10 tools and shuffled them !!
TOOL_REGISTRY = {
    "search_maps": {
        "short_desc": "Find locations, calculate routes, or get geographical coordinates.",
        "detailed_desc": "A geographic tool to find places, navigate maps, get driving directions, and locate specific points of interest like coffee shops or offices.",
        "full_schema": {"type": "function", "name": "search_maps", "parameters": {"location": "string"}}
    },
    "send_email": {
        "short_desc": "Send an email to a specific address.",
        "detailed_desc": "Communication tool to draft and send emails to users, clients, or internal team members using SMTP protocols.",
        "full_schema": {"type": "function", "name": "send_email", "parameters": {"to": "string", "body": "string"}}
    },
    "get_weather": {
        "short_desc": "Fetch current weather conditions for a location.",
        "detailed_desc": "Retrieves real-time weather data including temperature, humidity, wind speed, and forecasts for any city or coordinates.",
        "full_schema": {"type": "function", "name": "get_weather", "parameters": {"location": "string", "units": "string"}}
    },
    "translate_text": {
        "short_desc": "Translate text between languages.",
        "detailed_desc": "Uses NLP translation APIs to convert text from one language to another, supporting 100+ languages with dialect awareness.",
        "full_schema": {"type": "function","name": "translate_text","parameters": {"text": "string","source_lang": "string","target_lang": "string"}}
    },
    "create_calendar_event": {
        "short_desc": "Create a new event on a calendar.",
        "detailed_desc": "Adds a scheduled event to Google Calendar or Outlook with title, time, attendees, and optional video conferencing links.",
        "full_schema": {"type": "function","name": "create_calendar_event","parameters": {"title": "string","start_time": "string","end_time": "string","attendees": "list"}}
    },
    "get_stock_price": {
        "short_desc": "Fetch current stock prices and market data.",
        "detailed_desc": "Financial tool used to retrieve real-time trading data, stock tickers, equity prices, and market capitalization for public companies.",
        "full_schema": {"type": "function", "name": "get_stock_price", "parameters": {"ticker": "string"}}
    },
    "search_web": {
        "short_desc": "Search the internet for up-to-date information.",
        "detailed_desc": "Performs a web search via a search engine API and returns ranked results with titles, URLs, and snippets.",
        "full_schema": {"type": "function","name": "search_web","parameters": {"query": "string","num_results": "integer"}}
    },
    "run_code": {
        "short_desc": "Execute a code snippet in a sandboxed environment.",
        "detailed_desc": "Runs Python, JavaScript, or Bash code in an isolated runtime and returns stdout, stderr, and exit codes.",
        "full_schema": {"type": "function","name": "run_code","parameters": {"language": "string","code": "string"}}
    },
    "read_file": {
        "short_desc": "Read the contents of a file from a path or URL.",
        "detailed_desc": "Opens and returns the text content of local files or remote URLs, supporting txt, csv, json, and markdown formats.",
        "full_schema": {"type": "function","name": "read_file","parameters": {"path": "string"}}
    },
    "write_file": {
        "short_desc": "Write or overwrite content to a file.",
        "detailed_desc": "Creates or overwrites a file at a given path with specified text content, useful for saving outputs or generated data.",
        "full_schema": {"type": "function","name": "write_file","parameters": {"path": "string","content": "string"}}
    },
}

# Math Helper: Cosine Similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Stage 1: Embed the Tools (Mocking the Vector DB setup)
def build_vector_db():
    print("-> Embedding tools into memory...")
    for name, data in TOOL_REGISTRY.items():
        response = client.models.embed_content(
            model='gemini-embedding-001',
            contents=data["detailed_desc"]
        )
        data["embedding"] = response.embeddings[0].values
    return TOOL_REGISTRY

# Stage 2 & 3: The JIT Router Pipeline
def run_jit_protocol(user_query, top_k=5):
    print(f"\n[USER QUERY]: {user_query}")
    
    # 1. Embed the query
    query_embed = client.models.embed_content(
        model='gemini-embedding-001', 
        contents=user_query
    ).embeddings[0].values

    # 2. Vector Search (Coarse Filter)
    scores = []
    for name, data in TOOL_REGISTRY.items():
        score = cosine_similarity(query_embed, data["embedding"])
        scores.append((score, name, data["short_desc"]))
    
    # Sort and grab Top K
    scores.sort(reverse=True, key=lambda x: x[0])
    top_tools = scores[:top_k]
    
    print(f"-> Vector DB Found Top {top_k}: {[t[1] for t in top_tools]}")

    # 3. SLM Semantic Routing (Fine Filter using Gemini Flash)
    tools_context = "\n".join([f"- {name}: {desc}" for _, name, desc in top_tools])
    
    routing_prompt = f"""
    You are an intelligent tool router. Look at the user query and the available tools.
    Return a JSON list containing ONLY the exact names of the tools strictly required.
    
    Available Tools:
    {tools_context}
    
    User Query: {user_query}
    """

    print("-> SLM (Gemini 3 Flash) evaluating true intent...")
    slm_response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=routing_prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )

    selected_tools = json.loads(slm_response.text)
    print(f"-> SLM Selected: {selected_tools}")

    # 4. Final Injection Payload Assembly
    final_schemas = [TOOL_REGISTRY[tool]["full_schema"] for tool in selected_tools]
    print("\n[SUCCESS] Final Payload ready for the Heavy LLM:")
    print(json.dumps(final_schemas, indent=2))
    
    return final_schemas

# --- Execution ---
if __name__ == "__main__":
    build_vector_db()
    # Test a tricky query that might confuse a dumb search
    run_jit_protocol("What is the current trading price of Apple, and email it to my CFO.")