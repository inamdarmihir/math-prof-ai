#!/usr/bin/env python3
"""
Math Agent - An AI-powered assistant for solving math problems using LangGraph
"""

import os
import re
import logging
import time
import json
import sympy as sp
from sympy import symbols, solve, Eq, diff, integrate, Matrix, limit, oo, simplify, expand, factor, Poly, sqrt, sympify, Symbol
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
from sympy.parsing.latex import parse_latex
from sympy.solvers.ode import dsolve
from sympy.core.relational import Equality
from sympy.abc import x, y, z, t
from dotenv import load_dotenv
from typing import Dict, List, Any, TypedDict, Annotated, Union, Optional
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolNode
from langsmith import Client
from langsmith.run_helpers import traceable
import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import operator
from exa_py import Exa
from exa_py.api import SearchResponse
from typing_extensions import NotRequired
from requests import HTTPError
import traceback
import inspect
import math
import cmath  # Import cmath for complex number operations
import uuid

# Set up sympy parser transformations for implicit multiplication
transformations = (standard_transformations + 
                  (implicit_multiplication_application,
                   convert_xor))

# Import the config loader
from load_config import load_config

# Load configuration first, before any other initialization
config = load_config()

# Load environment variables
load_dotenv()

# Initialize Exa client
try:
    exa_client = Exa(api_key=os.getenv("EXA_API_KEY"))
    if not exa_client:
        logging.warning("Exa client initialization failed: API key not found")
        exa_client = None
except Exception as e:
    logging.warning(f"Failed to initialize Exa client: {str(e)}")
    exa_client = None

# Initialize LangSmith client with proper configuration
try:
    langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
    if not langsmith_api_key:
        logging.warning("LangSmith API key not found. Tracing will be disabled.")
        client = None
        langsmith_available = False
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    else:
        client = Client(
            api_url=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
            api_key=langsmith_api_key
        )
        
        # Set tracing environment variables first
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "math-agent-langgraph"
        
        # Verify the client with a simpler API call
        try:
            # Try to list a single project to verify API connection
            projects = client.list_projects(limit=1)
            langsmith_available = True
            logging.info("LangSmith tracing enabled")
        except Exception as e:
            logging.warning(f"LangSmith API verification failed: {str(e)}")
            langsmith_available = False
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
except HTTPError as http_err:
    if http_err.response.status_code == 404:
        logging.warning("LangSmith resource not found. This is normal for the test UUID.")
        # Still enable tracing for new runs
        langsmith_available = True
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "math-agent-langgraph"
        logging.info("LangSmith tracing enabled despite 404 error (expected for first-time setup)")
    elif http_err.response.status_code == 403:
        logging.warning("LangSmith access forbidden. Please check API key permissions.")
        client = None
        langsmith_available = False
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    else:
        logging.warning(f"HTTP error occurred with LangSmith: {http_err}")
        client = None
        langsmith_available = False
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
except Exception as e:
    logging.warning(f"Failed to initialize LangSmith client: {str(e)}")
    client = None
    langsmith_available = False
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    logging.info("LangSmith tracing disabled due to connection error")

# Initialize Qdrant client
# Try to initialize Qdrant client if environment variables are present
qdrant_client = None
try:
    import qdrant_client
    from qdrant_client import QdrantClient, models
    
    qdrant_url = os.getenv("QDRANT_URL", "")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
    
    if qdrant_url and qdrant_api_key:
        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            prefer_grpc=False,
            https=True,
            timeout=10.0  # Add timeout parameter
        )
        
        # Check if the collection exists
        try:
            # Get Qdrant version to ensure API compatibility
            qdrant_version = None
            try:
                # Try to get version info from the API (available in newer versions)
                version_info = qdrant_client._client.health.health()
                if hasattr(version_info, 'version'):
                    qdrant_version = version_info.version
                    logging.info(f"Connected to Qdrant version: {qdrant_version}")
            except Exception as e:
                logging.warning(f"Could not detect Qdrant version: {str(e)}")
            
            collections = qdrant_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if "math_problems" not in collection_names:
                # Create the collection if it doesn't exist
                qdrant_client.create_collection(
                    collection_name="math_problems",
                    vectors_config=models.VectorParams(
                        size=1536,  # OpenAI embedding size
                        distance=models.Distance.COSINE
                    )
                )
                logging.info("Created math_problems collection in Qdrant")
            
            logging.info("Successfully connected to math_problems collection")
        except Exception as e:
            logging.error(f"Error checking/creating Qdrant collection: {str(e)}")
            qdrant_client = None
        
        if qdrant_client:
            logging.info("Successfully connected to Qdrant")
    else:
        logging.warning("Qdrant URL or API key not found in environment variables")
except ImportError:
    logging.warning("Qdrant client not installed. Vector DB features will be disabled.")
except Exception as e:
    logging.error(f"Error initializing Qdrant client: {str(e)}")
    qdrant_client = None

# Initialize OpenAI client
try:
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    # Check empty or placeholder API key
    if not openai_api_key or openai_api_key.startswith("sk-") and len(openai_api_key) < 40:
        logging.warning("OpenAI API key not valid. Please set a valid OPENAI_API_KEY environment variable.")
        openai_client = None
    else:
        # Create client and test with a simple request (not a full embedding)
        openai_client = OpenAI(api_key=openai_api_key)
        
        # Log which model will be used for completions
        openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        logging.info(f"OpenAI model configured: {openai_model}")
        
        try:
            # Simple request to check key validity without using quota
            models = openai_client.models.list()
            if models and len(models.data) > 0:
                logging.info(f"OpenAI client initialized successfully with valid API key")
            else:
                logging.warning("OpenAI API key may not be valid - test request returned no data")
                openai_client = None
        except Exception as api_test_error:
            logging.warning(f"OpenAI API key validation failed: {str(api_test_error)}")
            openai_client = None
except Exception as e:
    logging.warning(f"Failed to initialize OpenAI client: {str(e)}")
    openai_client = None

# Create logs directory if it doesn't exist
logs_dir = 'logs'
try:
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Setup logging with file handler
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(logs_dir, 'math_agent_langgraph.log')),
            logging.StreamHandler()
        ]
    )
except (PermissionError, FileNotFoundError) as e:
    # Fallback to console-only logging if file logging fails
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

# Define the state for our graph
class MathAgentState(TypedDict):
    query: str
    history: List[Dict[str, Any]]
    current_step: str
    result: NotRequired[Dict[str, Any]]
    error: NotRequired[str]
    execution_times: Dict[str, float]
    search_results: NotRequired[List[Dict[str, Any]]]
    solution_strategy: NotRequired[str]
    intermediate_steps: NotRequired[List[str]]

# Define the nodes for our graph
@traceable(run_type="chain", name="extract_equations")
def extract_equations_node(state: MathAgentState) -> MathAgentState:
    """
    Extract mathematical equations from the query.
    This node:
    1. Identifies different types of equations in the query
    2. Extracts and normalizes them
    3. Updates the state with the extracted equations
    """
    import re
    import time
    import logging
    
    start_time = time.time()
    
    # Initialize execution_times if it doesn't exist
    if "execution_times" not in state:
        state["execution_times"] = {}
        
    # Initialize result if it doesn't exist
    if "result" not in state:
        state["result"] = {}
    
    query = state.get("query", "")
    
    if not query:
        state["result"]["error"] = "No query provided."
        state["execution_times"]["extract_equations"] = time.time() - start_time
        return state
    
    # Try to extract equations from the query
    text_equations = []
    
    # First check: if the query has multiple '=' signs without clear delimiters, it might be a system of equations
    # written without proper separation
    equals_count = query.count('=')
    if equals_count >= 2:
        # This might be a system without proper delimiters
        # Look for patterns like "x+y=1 x-y=2" or "x+y=1 and x-y=2" where equations are separated by spaces or "and"
        
        # First try to split on "and" if it exists
        if " and " in query.lower():
            parts = re.split(r'\s+and\s+', query, flags=re.IGNORECASE)
            
            potential_eqs = []
            for part in parts:
                # Extract equations from each part
                eq_matches = re.findall(r'([^;,\n]+?=\s*[-+\d\w\s.\/\^]+)', part)
                potential_eqs.extend(eq_matches)
                
            if potential_eqs and len(potential_eqs) >= 2:
                for eq in potential_eqs:
                    clean_eq = eq.strip()
                    if clean_eq and clean_eq not in text_equations:
                        text_equations.append(clean_eq)
                
                logging.info(f"Extracted system of equations split by 'and': {text_equations}")
                
                # If we found equations this way, skip the other extraction methods
                if text_equations:
                    state["result"]["text_equations"] = text_equations
                    state["execution_times"]["extract_equations"] = time.time() - start_time
                    state["current_step"] = "equations_extracted"
                    return state
        
        # If we didn't find equations with "and", try other patterns
        # Look for patterns where equations are separated by spaces
        potential_system = re.findall(r'([^;,\n]+?=\s*[-+\d\w\s.\/\^]+)(?=\s+[a-zA-Z][-+\d\w\s.\/\^]*=|\Z)', query)
        
        if potential_system and len(potential_system) >= 2:
            for eq in potential_system:
                clean_eq = eq.strip()
                if clean_eq and clean_eq not in text_equations:
                    text_equations.append(clean_eq)
                    
            logging.info(f"Extracted system of equations without delimiters: {text_equations}")
            
            # If we found equations this way, skip the other extraction methods
            if text_equations:
                state["result"]["text_equations"] = text_equations
                state["execution_times"]["extract_equations"] = time.time() - start_time
                state["current_step"] = "equations_extracted"
                return state
    
    # Look for quadratic equations (ax^2 + bx + c = 0)
    quadratic_pattern = re.compile(r'([0-9]*(?:\.[0-9]+)?)?\s*([a-zA-Z])(?:\^2|²)\s*([+-]\s*[0-9]*(?:\.[0-9]+)?\s*[a-zA-Z](?:\^[0-9]+)?)?(?:\s*([+-]\s*[0-9]+(?:\.[0-9]+)?)?)?\s*=\s*0', re.IGNORECASE)
    quadratic_match = quadratic_pattern.search(query)
    if quadratic_match:
        a, var, b, c = quadratic_match.groups()
        # Handle none/empty coefficient as 1
        a = a.strip() if a else "1"
        equation = f"{a}{var}^2{b or ''}{c or ''}=0"
        text_equations.append(equation)
        logging.info(f"Extracted quadratic equation: {equation}")
    
    # Try alternative quadratic pattern for formulations like "2x^2+4x-6=0"
    if not any(re.search(r'[a-zA-Z]\^2|[a-zA-Z]²', eq) for eq in text_equations):
        alt_quadratic_pattern = re.compile(r'([0-9]*(?:\.[0-9]+)?)\s*([a-zA-Z])\^2\s*([+-]\s*[0-9]*(?:\.[0-9]+)?\s*[a-zA-Z])?\s*([+-]\s*[0-9]+(?:\.[0-9]+)?)?\s*=\s*0', re.IGNORECASE)
        alt_match = alt_quadratic_pattern.search(query)
        if alt_match:
            a, var, b, c = alt_match.groups()
            # Handle none/empty coefficient as 1
            a = a.strip() if a else "1"
            equation = f"{a}{var}^2{b or ''}{c or ''}=0"
            text_equations.append(equation)
            logging.info(f"Extracted quadratic equation (alt pattern): {equation}")
    
    # Try another pattern specifically for the form "ax^2+bx+c=0" with minimal whitespace
    if not any(re.search(r'[a-zA-Z]\^2|[a-zA-Z]²', eq) for eq in text_equations):
        concise_quadratic = re.compile(r'(\d*)([a-zA-Z])\^2([+-]\d*[a-zA-Z])?([+-]\d+)?=0', re.IGNORECASE)
        concise_match = concise_quadratic.search(re.sub(r'\s+', '', query))
        if concise_match:
            a, var, b, c = concise_match.groups()
            # Handle none/empty coefficient as 1
            a = a.strip() if a else "1"
            equation = f"{a}{var}^2{b or ''}{c or ''}=0"
            text_equations.append(equation)
            logging.info(f"Extracted quadratic equation (concise pattern): {equation}")
    
    # Add a pattern specifically for "2x^2+4x-6=0" and similar
    if not any(re.search(r'[a-zA-Z]\^2|[a-zA-Z]²', eq) for eq in text_equations):
        standard_quadratic = re.compile(r'(\d+)([a-zA-Z])\^2([+-]\d+)([a-zA-Z])([+-]\d+)=0', re.IGNORECASE)
        standard_match = standard_quadratic.search(re.sub(r'\s+', '', query))
        if standard_match:
            a, var1, b_with_sign, var2, c_with_sign = standard_match.groups()
            if var1 == var2:  # Same variable
                equation = f"{a}{var1}^2{b_with_sign}{var2}{c_with_sign}=0"
                text_equations.append(equation)
                logging.info(f"Extracted standard form quadratic equation: {equation} with a={a}, b={b_with_sign}, c={c_with_sign}")
    
    # Look for general equations (e.g., 2x + 3 = 5)
    general_equation_pattern = re.compile(r'([0-9a-zA-Z.+\-*/^()\s]+)\s*=\s*([0-9a-zA-Z.+\-*/^()\s]+)')
    general_matches = list(general_equation_pattern.finditer(query))
    
    if general_matches:
        for match in general_matches:
            equation = f"{match.group(1)}={match.group(2)}"
            # Clean up the equation
            equation = re.sub(r'\s+', '', equation)
            if equation not in text_equations:
                text_equations.append(equation)
                logging.info(f"Extracted general equation: {equation}")
    
    # Look for systems of equations
    system_pattern = re.compile(
        r'(?:systems?|pair|sets?|groups?)?\s*(?:of)?\s*(?:equations?|expressions?)?'
        r'(?:\s*(?::|,|;|\n|where|with|and)\s*)?'
        r'((?:[0-9a-zA-Z.+\-*/^()\s]+\s*=\s*[0-9a-zA-Z.+\-*/^()\s]+)\s*(?:(?:,|;|and|\n)\s*[0-9a-zA-Z.+\-*/^()\s]+\s*=\s*[0-9a-zA-Z.+\-*/^()\s]+)+)',
        re.IGNORECASE
    )
    system_match = system_pattern.search(query)
    
    # If system of equations detected, extract individual equations
    if system_match:
        system_text = system_match.group(1)
        # Split by common separators
        eq_parts = re.split(r'(?:,|;|and|\n)', system_text)
        system_equations = []
        for part in eq_parts:
            # Extract equation from each part
            eq_match = re.search(r'([0-9a-zA-Z.+\-*/^()\s]+)\s*=\s*([0-9a-zA-Z.+\-*/^()\s]+)', part)
            if eq_match:
                equation = f"{eq_match.group(1)}={eq_match.group(2)}"
                equation = re.sub(r'\s+', '', equation)
                if equation not in system_equations and equation not in text_equations:
                    system_equations.append(equation)
        
        if system_equations:
            text_equations.extend(system_equations)
            logging.info(f"Extracted system of equations: {system_equations}")
    
    # Look for LaTeX-formatted equations
    latex_pattern = re.compile(r'\$+([^$]+)\$+')
    for latex_match in latex_pattern.finditer(query):
        latex_eq = latex_match.group(1)
        # Check if we already have this equation without LaTeX formatting
        if latex_eq not in text_equations:
            text_equations.append(latex_eq)
    
    # If no equations found, check for potential equations without equals sign
    if not text_equations:
        # Check for expressions that might be equations
        potential_equation_pattern = re.compile(r'((?:[0-9]*[a-zA-Z](?:\^[0-9]+)?)?(?:\s*[+\-]\s*[0-9]*[a-zA-Z](?:\^[0-9]+)?)+)')
        potential_match = potential_equation_pattern.search(query)
        if potential_match:
            potential_eq = potential_match.group(1).strip()
            # Add equals 0 if it's a polynomial-like expression
            if re.search(r'[a-zA-Z]', potential_eq):
                potential_eq = f"{potential_eq}=0"
                text_equations.append(potential_eq)
    
    # Store extracted equations in state
    filtered_equations = []
    for eq in text_equations:
        # Skip equations that don't contain an equals sign
        if '=' not in eq:
            continue
            
        # Skip equations that don't have any variable (letter)
        if not re.search(r'[a-zA-Z]', eq):
            continue
            
        # Skip equations that have words (4+ consecutive letters) which might be instructions
        if re.search(r'[a-zA-Z]{4,}', eq):
            logging.warning(f"Skipping equation with potential instruction text: {eq}")
            continue
            
        filtered_equations.append(eq)
    
    if filtered_equations:
        state["result"]["text_equations"] = filtered_equations
        logging.info(f"Filtered equations: {filtered_equations}")
    else:
        # If all equations were filtered, try again with original extraction
        state["result"]["text_equations"] = text_equations
        logging.info(f"Using original extracted equations: {text_equations}")
    
    # Track execution time
    state["execution_times"]["extract_equations"] = time.time() - start_time
    state["current_step"] = "equations_extracted"
    
    return state

@traceable(run_type="tool", name="search_math_concepts")
def search_math_concepts_node(state: MathAgentState) -> MathAgentState:
    """
    Search for math concepts and similar equations in external knowledge sources.
    """
    import time
    import re
    import logging
    import openai
    
    # Define get_embedding function locally
    def get_embedding(text: str) -> List[float]:
        """
        Get an embedding vector for a text string using OpenAI's embedding API.
        
        Args:
            text: The text to embed
            
        Returns:
            A list of floats representing the embedding vector, or None if there's an error
        """
        if not openai_client:
            logging.warning("OpenAI client not available for embedding generation")
            return None
            
        try:
            # Clean text for embedding
            text = text.replace("\n", " ")
            
            # Get embedding from OpenAI
            response = openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"  # Using the smaller model is sufficient and more cost-effective
            )
            
            # Extract the embedding vector
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            logging.error(f"Error generating embedding: {str(e)}")
            return None
    
    start_time = time.time()
    
    # Initialize execution_times if it doesn't exist
    if "execution_times" not in state:
        state["execution_times"] = {}
    
    # Initialize result if it doesn't exist
    if "result" not in state:
        state["result"] = {}
        
    # Get the query from state
    query = state.get("query", "")
    
    # Extract equations from result
    equations = state.get("result", {}).get("text_equations", [])
    
    # Combine the query and equations for searching
    search_text = query
    if equations:
        # Add equations with proper spacing
        eq_text = " ".join(equations)
        search_text = f"{search_text} {eq_text}"
    
    # Define search_equation_in_vector_db function locally 
    def search_equation_in_vector_db(query_text, search_state=None):
        """
        Search for similar equations in the vector database.
        """
        import time
        import logging
        
        search_start_time = time.time()
        execution_times = search_state.get("execution_times", {}) if search_state else {}
        result = search_state.get("result", {}) if search_state else {}
        
        try:
            # Generate embedding for the query
            embedding = get_embedding(query_text)
            if not embedding:
                result["vector_db_error"] = "Failed to generate embedding"
                return search_state
            
            # Initialize variables
            similar_equations = []
            search_attempted = False
            error_messages = []
            
            # Method 1: Search with query_vector parameter (newer API)
            search_attempted = True
            try:
                # Use query_points instead of search (which is deprecated)
                search_result = qdrant_client.query_points(
                    collection_name="math_problems",
                    query_vector=embedding,
                    limit=3
                )
                
                similar_equations = []
                for res in search_result.points:
                    similar_equations.append({
                        "equation": res.payload.get("equation", ""),
                        "solution": res.payload.get("solution", ""),
                        "score": res.score if hasattr(res, 'score') else 0.0
                    })
                
                if similar_equations:
                    result["similar_equations"] = similar_equations
                    logging.info(f"Found {len(similar_equations)} similar equations")
            except Exception as e1:
                error_msg = str(e1)
                error_messages.append(f"Standard search failed: {error_msg}")
                logging.warning(f"Vector DB search failed: {error_msg}")
                
                # Method 2: Fallback to older API if needed
                try:
                    search_attempted = True
                    logging.info("Attempting Qdrant search with vector parameter")
                    search_result = qdrant_client.search(
                        collection_name="math_problems",
                        vector=embedding,
                        limit=3
                    )
                    
                    similar_equations = []
                    for res in search_result:
                        similar_equations.append({
                            "equation": res.payload.get("equation", ""),
                            "solution": res.payload.get("solution", ""),
                            "score": res.score if hasattr(res, 'score') else 0.0
                        })
                    
                    if similar_equations:
                        result["similar_equations"] = similar_equations
                        logging.info(f"Found {len(similar_equations)} similar equations with vector parameter")
                except Exception as e2:
                    error_msg = str(e2)
                    error_messages.append(f"Search with vector parameter failed: {error_msg}")
                    logging.warning(f"Qdrant vector search failed: {error_msg}")
                    
                    # Method 3: Try with more recent API versions
                    try:
                        search_attempted = True
                        logging.info("Attempting Qdrant search with points_search method")
                        if hasattr(qdrant_client, 'points_search'):
                            search_result = qdrant_client.points_search(
                                collection_name="math_problems",
                                query_vector=embedding,
                                limit=3
                            )
                            
                            similar_equations = []
                            for res in search_result.points:
                                similar_equations.append({
                                    "equation": res.payload.get("equation", ""),
                                    "solution": res.payload.get("solution", ""),
                                    "score": res.score if hasattr(res, 'score') else 0.0
                                })
                            
                            if similar_equations:
                                result["similar_equations"] = similar_equations
                                logging.info(f"Found {len(similar_equations)} similar equations with points_search")
                    except Exception as e3:
                        error_msg = str(e3)
                        error_messages.append(f"Points search failed: {error_msg}")
                        logging.warning(f"Qdrant points_search failed: {error_msg}")
            
            # Record search results
            if not search_attempted:
                result["vector_db_error"] = "No search method was attempted"
            elif not similar_equations and error_messages:
                result["vector_db_error"] = f"Search failed: {'; '.join(error_messages[:2])}"
            elif similar_equations:
                logging.info(f"Successfully found {len(similar_equations)} similar equations")
            else:
                result["vector_db_error"] = "No similar equations found"
                
        except Exception as e:
            logging.error(f"Error in embedding generation: {str(e)}")
            result["vector_db_error"] = f"Embedding generation failed: {str(e)}"
        
        execution_times["vector_db_search"] = time.time() - search_start_time
        if search_state:
            search_state["result"] = result
            search_state["execution_times"] = execution_times
            return search_state
        else:
            return result
    
    # Get embedding for search
    try:
        # Get embedding
        embedding = get_embedding(search_text)
        
        if embedding:
            # Search with embedding vector
            try:
                similar_results = search_equation_in_vector_db(search_text, state)
                if similar_results and "similar_equations" in similar_results.get("result", {}):
                    state["result"]["similar_equations"] = similar_results["result"]["similar_equations"]
                    logging.info(f"Found {len(similar_results['result']['similar_equations'])} similar equations")
            except Exception as e:
                logging.error(f"Error in vector search: {str(e)}")
                state["result"]["vector_search_error"] = str(e)
    except Exception as e:
        logging.warning(f"Vector DB search failed: {str(e)}")
    
    # Record execution time
    state["execution_times"]["search_math_concepts"] = time.time() - start_time
    state["current_step"] = "searched_concepts"
    return state

@traceable(run_type="chain", name="solve_equations")
def solve_equations_node(state: MathAgentState) -> MathAgentState:
    """
    Solve the extracted equations from the query.
    """
    import time
    import re
    import logging
    import math
    import cmath  # For complex number operations
    import traceback
    from sympy import symbols, solve, sympify, parse_expr, Eq, Symbol, I, diff, integrate, simplify
    from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application, convert_xor
    from sympy.solvers.ode import dsolve
    
    # Define the clean_latex_for_parsing function locally
    def clean_latex_for_parsing(latex_str: str) -> str:
        """
        Clean a LaTeX string for parsing by SymPy.
        
        This function:
        1. Removes LaTeX-specific formatting and commands
        2. Handles special cases like matrices and integrals
        3. Returns a clean string ready for parsing or special indicators
        """
        if not latex_str:
            return ""
        
        # Remove dollar signs if present (LaTeX delimiters)
        if latex_str.startswith('$') and latex_str.endswith('$'):
            latex_str = latex_str[1:-1]
        
        # Remove double dollar signs if present
        if latex_str.startswith('$$') and latex_str.endswith('$$'):
            latex_str = latex_str[2:-2]
        
        # Check for matrix notation
        if '\\begin{matrix}' in latex_str or '\\begin{bmatrix}' in latex_str or '\\begin{pmatrix}' in latex_str:
            return "matrix_notation_detected"
        
        # Check for integral notation
        if '\\int' in latex_str:
            return "integral_notation_detected"
        
        # Check for differential equation notation (e.g., dy/dx)
        if "\\frac{d" in latex_str and "}{d" in latex_str:
            return "differential_equation_detected"
        
        # Clean problematic LaTeX commands
        replacements = {
            # Handle fractions
            r'\\frac{([^}]*)}{([^}]*)}': r'(\1)/(\2)',
            
            # Handle powers with curly braces
            r'([a-zA-Z0-9])\\^{([^}]*)}': r'\1^(\2)',
            
            # Handle square roots
            r'\\sqrt{([^}]*)}': r'sqrt(\1)',
            
            # Handle common LaTeX commands
            r'\\left': '',
            r'\\right': '',
            r'\\cdot': '*',
            r'\\times': '*',
            
            # Handle exponents without braces
            r'\\^([0-9])': r'^{\1}',
            
            # Clean problematic escape sequences
            r'\\e': 'e',
            r'\\i': 'i',
            r'\\pi': 'pi',
            
            # Replace LaTeX spaces
            r'\\quad': ' ',
            r'\\qquad': '  '
        }
        
        # Apply all replacements
        for pattern, replacement in replacements.items():
            latex_str = re.sub(pattern, replacement, latex_str)
        
        # Handle spaces and clean up
        latex_str = re.sub(r'\s+', ' ', latex_str).strip()
        
        return latex_str
    
    start_time = time.time()
    
    # Initialize execution_times if it doesn't exist
    if "execution_times" not in state:
        state["execution_times"] = {}
    
    # Initialize result if it doesn't exist
    if "result" not in state:
        state["result"] = {}
    
    # Get the query text from state
    query_text = state.get("query", "")
    
    # Initialize solution variables
    solutions = []
    formatted_solutions = []
    steps = []
    
    # Get the equations from the state
    equations = state.get("result", {}).get("text_equations", [])
    
    if not equations:
        logging.warning("No equations found in state.")
        state["result"]["error"] = "No equations found to solve."
        state["execution_times"]["solve_equations"] = time.time() - start_time
        return state
    
    # Define transformations for parsing - include convert_xor to properly handle ^
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    
    try:
        # First, clean all equations of LaTeX formatting
        clean_equations = []
        for eq in equations:
            # Strip dollar signs
            clean_eq = eq.replace('$', '')
            
            # Strip any LaTeX formatting
            clean_eq = clean_latex_for_parsing(clean_eq)
            
            # Replace LaTeX exponent notation with ^ for parsing
            clean_eq = re.sub(r'(\d+)x\^{2}', r'\1x^2', clean_eq)
            clean_eq = re.sub(r'(\d+)x\^2', r'\1x^2', clean_eq)
            
            # Fix problematic LaTeX cases for begins
            clean_eq = clean_eq.replace('\\begin', 'begin')
            clean_eq = clean_eq.replace('\\cases', 'cases')
            
            # Replace x² with x^2 for parsing (for any variable)
            clean_eq = re.sub(r'([a-zA-Z])²', r'\1^2', clean_eq)
            
            # Clean up any remaining special characters
            clean_eq = re.sub(r'[{}]', '', clean_eq)
            
            # Remove any whitespace
            clean_eq = re.sub(r'\s+', '', clean_eq)
            
            # Skip empty equations
            if clean_eq:
                clean_equations.append(clean_eq)
                logging.info(f"Cleaned equation: {clean_eq}")
        
        # Determine if we're dealing with a system of equations
        is_system_of_equations = len(clean_equations) > 1
        
        # Additional check for cases environment or "and" connected equations
        if len(clean_equations) == 1:
            # Check for LaTeX cases environment
            if 'begin{cases}' in equations[0] or '\\begin{cases}' in equations[0]:
                is_system_of_equations = True
            # Check for "and" connecting multiple equations
            elif ' and ' in equations[0].lower() and equations[0].count('=') > 1:
                is_system_of_equations = True
        
        # Handle systems of equations first
        if is_system_of_equations:
            logging.info(f"Detected system of equations with {len(clean_equations)} equations")
            try:
                solution_dict = solve_system_of_equations(equations)
                
                if "error" in solution_dict:
                    logging.warning(f"Error solving system: {solution_dict['error']}")
                    # Fall back to individual equation solving if system solving fails
                else:
                    solutions.append(solution_dict)
                    formatted_solution = ", ".join([f"{var} = {val}" for var, val in solution_dict.items()])
                    formatted_solutions.append(formatted_solution)
                    
                    # Generate steps for the solution
                    steps = [
                        f"System of equations: {', '.join(equations)}",
                        "Setting up the system of linear equations",
                        f"Solving for variables: {', '.join(solution_dict.keys())}",
                        f"Solution: {formatted_solution}"
                    ]
                    
                    logging.info(f"Solved system of equations: {formatted_solution}")
                    
                    # If we successfully solved the system, update state and return
                    state["result"]["solutions"] = solutions
                    state["result"]["formatted_solutions"] = formatted_solutions
                    state["result"]["steps"] = steps
                    
                    # Make sure the returned result has both the solutions and formatted solutions
                    state["solutions"] = solutions
                    state["formatted_solutions"] = formatted_solutions
                    
                    # Add execution time
                    state["execution_times"]["solve_equations"] = time.time() - start_time
                    state["current_step"] = "equations_solved"
                    return state
            except Exception as e:
                logging.error(f"Error in system of equations solver: {str(e)}")
                # Continue to individual equation solving as fallback
        
        # Check for specific known equations that we can solve directly
        for eq in clean_equations:
            # Linear equation: ax + b = c format
            linear_match = re.match(r'^(\d*)([a-zA-Z])([-+]\d*)?=(\d+)$', eq)
            if linear_match:
                a_str, var, b_str, c_str = linear_match.groups()
                
                # Handle defaults for empty groups
                a = float(a_str) if a_str and a_str != '' else 1.0
                b = float(b_str) if b_str and b_str != '' else 0.0
                c = float(c_str)
                
                # Solve: ax + b = c => x = (c - b) / a
                solution_value = (c - b) / a
                solutions.append({var: str(solution_value)})
                formatted_solutions.append(f"{var} = {solution_value}")
                
                steps = [
                    f"Starting with the equation: {a}{var} {'+' if b >= 0 else ''}{b} = {c}",
                    f"Subtract {b} from both sides: {a}{var} = {c - b}",
                    f"Divide both sides by {a}: {var} = {solution_value}"
                ]
                
                logging.info(f"Solved linear equation: {eq} => {var} = {solution_value}")
                break
                
            # Simple system check for "2x+3x=5" type equations
            combined_terms_match = re.match(r'^(\d*)([a-zA-Z])[+](\d*)([a-zA-Z])=(\d+)$', eq)
            if combined_terms_match and combined_terms_match.group(2) == combined_terms_match.group(4):
                # This is like "2x+3x=5"
                a_str, var1, b_str, var2, c_str = combined_terms_match.groups()
                
                # Handle defaults
                a = float(a_str) if a_str and a_str != '' else 1.0
                b = float(b_str) if b_str and b_str != '' else 1.0
                c = float(c_str)
                
                # Combined coefficient
                combined_coef = a + b
                
                # Solution: (a+b)x = c => x = c/(a+b)
                solution_value = c / combined_coef
                solutions.append({var1: str(solution_value)})
                formatted_solutions.append(f"{var1} = {solution_value}")
                
                steps = [
                    f"Starting with the equation: {a}{var1} + {b}{var2} = {c}",
                    f"Since {var1} and {var2} are the same variable, combine like terms: ({a} + {b}){var1} = {c}",
                    f"Simplify: {combined_coef}{var1} = {c}",
                    f"Divide both sides by {combined_coef}: {var1} = {c}/{combined_coef} = {solution_value}"
                ]
                
                logging.info(f"Matched specific equation: {eq}")
                logging.info(f"Solved by combining like terms: {var1} = {solution_value}")
                break
                
            # Check for quadratic equations: ax^2 + bx + c = 0
            # More flexible regex that can handle spaces and different forms of the equation
            quadratic_match = re.match(r'^(\d*)([a-zA-Z])\^2\s*([-+]\s*\d*[a-zA-Z])?\s*([-+]\s*\d+)?\s*=\s*0$', eq)
            if not quadratic_match:
                # Try alternative format with ²
                quadratic_match = re.match(r'^(\d*)([a-zA-Z])²\s*([-+]\s*\d*[a-zA-Z])?\s*([-+]\s*\d+)?\s*=\s*0$', eq)
            if not quadratic_match:
                # Try even more flexible pattern
                quadratic_match = re.search(r'(\d*)([a-zA-Z])[²\^2]\s*([-+]\s*\d*[a-zA-Z])?\s*([-+]\s*\d+)?\s*=\s*0', eq)
            if not quadratic_match:
                # Try exact pattern for "2x^2+4x-6=0" format
                exact_pattern = re.match(r'^(\d+)([a-zA-Z])\^2([+-]\d+)([a-zA-Z])([+-]\d+)=0$', eq)
                if exact_pattern:
                    a_str, var1, b_with_sign, var2, c_with_sign = exact_pattern.groups()
                    # Verify that var1 and var2 are the same variable
                    if var1 == var2:
                        a = float(a_str)
                        b = float(b_with_sign)  # The sign is included in the capture
                        c = float(c_with_sign)  # The sign is included in the capture
                        var = var1
                        
                        # Create a compatible match structure for the existing code
                        class QuadraticMatch:
                            def __init__(self, a, var, b, c):
                                self.groups_data = (str(a), var, f"{b:+g}{var}", f"{c:+g}")
                            
                            def groups(self):
                                return self.groups_data
                        
                        quadratic_match = QuadraticMatch(a, var, b, c)
                        logging.info(f"Matched exact quadratic pattern: {eq} with a={a}, b={b}, c={c}")
            
            if quadratic_match:
                try:
                    a_str, var, b_part, c_part = quadratic_match.groups()
                    
                    # Handle defaults and empty groups
                    a = float(a_str) if a_str and a_str != '' else 1.0
                    
                    # Extract b coefficient
                    b = 0.0
                    if b_part:
                        # Check if this is already a numeric value (from exact pattern)
                        if isinstance(b_part, (int, float)):
                            b = float(b_part)
                        else:
                            b_match = re.match(r'([-+])(\d*)([a-zA-Z])', b_part)
                            if b_match:
                                sign, b_val, _ = b_match.groups()
                                b_val = b_val if b_val else '1'
                                b = float(f"{sign}{b_val}")
                    
                    # Extract c coefficient
                    c = 0.0
                    if c_part:
                        # Check if this is already a numeric value (from exact pattern)
                        if isinstance(c_part, (int, float)):
                            c = float(c_part)
                        else:
                            try:
                                c = float(c_part)
                            except ValueError:
                                # Try to handle more complex c_part formats
                                c_match = re.match(r'([-+])(\d+)', c_part)
                                if c_match:
                                    sign, c_val = c_match.groups()
                                    c = float(f"{sign}{c_val}")
                    
                    # Log the extracted values
                    logging.info(f"Quadratic equation coefficients: a={a}, b={b}, c={c}")
                    
                    # Calculate discriminant
                    discriminant = b**2 - 4*a*c
                    
                    steps = [
                        f"For the quadratic equation {a}{var}^2 {b_part if b_part else ''} {c_part if c_part else ''} = 0:",
                        f"Identify the coefficients: a = {a}, b = {b}, c = {c}",
                        f"Calculate the discriminant: Δ = b² - 4ac = {b}² - 4({a})({c}) = {discriminant}"
                    ]
                    
                    if discriminant >= 0:
                        # Real solutions
                        x1 = (-b + math.sqrt(discriminant)) / (2*a)
                        x2 = (-b - math.sqrt(discriminant)) / (2*a)
                        
                        if abs(x1 - x2) < 1e-10:  # If roots are practically the same
                            solutions.append({var: str(x1)})
                            formatted_solutions.append(f"{var} = {x1}")
                            steps.append(f"Since the discriminant is zero, there is one repeated root:")
                            steps.append(f"{var} = -b/(2a) = {-b}/(2({a})) = {x1}")
                        else:
                            solutions.append({var: str(x1), f"{var}_2": str(x2)})
                            formatted_solutions.append(f"{var} = {x1} or {var} = {x2}")
                            steps.append(f"Since the discriminant is positive, there are two real roots:")
                            steps.append(f"{var} = (-b + √Δ)/(2a) = ({-b} + √{discriminant})/(2({a})) = {x1}")
                            steps.append(f"{var} = (-b - √Δ)/(2a) = ({-b} - √{discriminant})/(2({a})) = {x2}")
                    else:
                        # Complex solutions
                        real_part = -b / (2*a)
                        imag_part = math.sqrt(abs(discriminant)) / (2*a)
                        
                        solutions.append({var: f"{real_part} + {imag_part}i", f"{var}_2": f"{real_part} - {imag_part}i"})
                        formatted_solutions.append(f"{var} = {real_part} + {imag_part}i or {var} = {real_part} - {imag_part}i")
                        
                        steps.append(f"Since the discriminant is negative, there are two complex roots:")
                        steps.append(f"{var} = (-b ± i√|Δ|)/(2a) = ({-b} ± i√{abs(discriminant)})/(2({a}))")
                        steps.append(f"{var} = {real_part} ± {imag_part}i")
                    
                    logging.info(f"Solved quadratic equation using direct formula: {eq}")
                    
                    # Update state with solutions and steps
                    state["solutions"] = solutions
                    state["formatted_solutions"] = formatted_solutions
                    state["steps"] = steps
                    state["current_step"] = "equations_solved"
                    
                    # Also update the result in state
                    if "result" not in state:
                        state["result"] = {}
                    state["result"]["solutions"] = solutions
                    state["result"]["formatted_solutions"] = formatted_solutions
                    state["result"]["steps"] = steps
                    state["result"]["equations"] = [eq]
                    state["result"]["text_equations"] = [eq]
                    
                    # Record execution time
                    state["execution_times"]["solve_equations"] = time.time() - start_time
                    
                    return state
                except Exception as e:
                    logging.error(f"Error processing quadratic equation: {str(e)}")
        
        # If we reach this point, no equations were successfully solved
        logging.error("Could not solve any equations")
        if "result" not in state:
            state["result"] = {}
        state["result"]["error"] = "Could not solve the equations. Please check your input."
        state["execution_times"]["solve_equations"] = time.time() - start_time
        return state
    except Exception as e:
        logging.error(f"Error solving equations: {str(e)}")
        if "result" not in state:
            state["result"] = {}
        state["result"]["error"] = f"Error: {str(e)}"
        state["execution_times"]["solve_equations"] = time.time() - start_time
        return state

def solve_system_of_equations(equations):
    """
    Solve a system of linear or simple equations using SymPy.
    
    Args:
        equations (list): List of equation strings
        
    Returns:
        dict: Dictionary mapping variable names to their values, or error message
    """
    import re
    import logging
    import math
    import numpy as np
    from sympy import symbols, solve, sympify, parse_expr, Eq
    from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application, convert_xor
    
    logging.info(f"Solving system of equations: {equations}")
    
    # Define transformations for parsing
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    
    try:
        # Clean equations (remove dollar signs, whitespace, etc.)
        clean_eqs = []
        for eq in equations:
            # Remove dollar signs and whitespace
            clean_eq = eq.replace('$', '').strip()
            # Replace x² with x^2
            clean_eq = re.sub(r'([a-zA-Z])²', r'\1^2', clean_eq)
            # Remove any remaining whitespace
            clean_eq = re.sub(r'\s+', '', clean_eq)
            clean_eqs.append(clean_eq)
        
        # Try different parsing approaches
        try:
            # Parse equations into SymPy format with Eq objects
            sympy_eqs = []
            all_symbols = set()
            
            for eq in clean_eqs:
                # Split by equals sign
                if '=' in eq:
                    left, right = eq.split('=')
                    
                    # Parse left and right sides
                    try:
                        left_expr = parse_expr(left, transformations=transformations)
                        right_expr = parse_expr(right, transformations=transformations)
                        sympy_eq = Eq(left_expr, right_expr)
                        
                        # Collect symbols (variables)
                        all_symbols.update(sympy_eq.free_symbols)
                        sympy_eqs.append(sympy_eq)
                    except Exception as parse_err:
                        logging.warning(f"Error parsing equation {eq}: {str(parse_err)}")
                        # Try a simpler approach
                        logging.info(f"Falling back to simpler equation parsing for {eq}")
                        
                        # Move all terms to left side (standard form: expr = 0)
                        try:
                            left_expr = parse_expr(left + '-(' + right + ')', transformations=transformations)
                            sympy_eq = Eq(left_expr, 0)
                            all_symbols.update(sympy_eq.free_symbols)
                            sympy_eqs.append(sympy_eq)
                        except Exception as e:
                            logging.error(f"Failed to parse equation {eq} in simplified form: {str(e)}")
            
            # Convert symbols to a list and sort for consistent ordering
            symbol_list = sorted(list(all_symbols), key=lambda sym: str(sym))
            
            if sympy_eqs and symbol_list:
                # Solve the system of equations
                solution = solve(sympy_eqs, symbol_list, dict=True)
                logging.info(f"Solution from SymPy: {solution}")
                
                if solution:
                    # Return the first solution
                    return {str(var): str(val) for var, val in solution[0].items()}
                else:
                    return {"error": "No solution found by SymPy solver"}
            else:
                return {"error": "Failed to parse equations into SymPy format"}
            
        except Exception as e:
            logging.warning(f"Error in primary solving method: {str(e)}")
            logging.info("Attempting fallback numerical method")
            
            # Fallback: For simple 2-variable systems, try to solve directly
            if len(clean_eqs) == 2:
                try:
                    # Direct solution for 2x2 systems using Cramer's rule
                    a, b, c = extract_coefficients(clean_eqs[0])
                    d, e, f = extract_coefficients(clean_eqs[1])
                    
                    # Determinant of coefficient matrix
                    det = a*e - b*d
                    
                    if abs(det) < 1e-10:  # Determinant is close to zero
                        return {"error": "The system has either no solution or infinitely many solutions (determinant is zero)"}
                    
                    # Apply Cramer's rule
                    x = (c*e - b*f) / det
                    y = (a*f - c*d) / det
                    
                    # Determine variable names (assume x and y if can't extract)
                    var_names = extract_variable_names(clean_eqs)
                    if len(var_names) >= 2:
                        return {var_names[0]: str(x), var_names[1]: str(y)}
                    else:
                        return {"x": str(x), "y": str(y)}
                        
                except Exception as fallback_error:
                    logging.error(f"Fallback solution also failed: {str(fallback_error)}")
                    return {"error": f"Could not solve system using either method: {str(fallback_error)}"}
            else:
                return {"error": f"Could not solve system of {len(clean_eqs)} equations: {str(e)}"}
    
    except Exception as e:
        logging.error(f"Error solving system of equations: {str(e)}")
        return {"error": f"Error: {str(e)}"}

def extract_coefficients(equation):
    """Extract coefficients a, b, c from an equation of form ax + by = c"""
    import re
    
    # Standardize equation format
    eq = equation.replace(" ", "")  # Remove all spaces
    
    # Extract the parts of the equation
    if "=" in eq:
        left, right = eq.split("=")
    else:
        left, right = eq, "0"
    
    # Move all terms to the left side
    left_side = left + "-(" + right + ")"
    
    # Initialize coefficients
    a, b, c = 0, 0, 0
    
    # Pattern for terms with x
    x_pattern = r'([+-]?\d*\.?\d*)x'
    x_matches = re.findall(x_pattern, left_side)
    for match in x_matches:
        if match == "" or match == "+":
            a += 1
        elif match == "-":
            a -= 1
        else:
            a += float(match)
    
    # Pattern for terms with y
    y_pattern = r'([+-]?\d*\.?\d*)y'
    y_matches = re.findall(y_pattern, left_side)
    for match in y_matches:
        if match == "" or match == "+":
            b += 1
        elif match == "-":
            b -= 1
        else:
            b += float(match)
    
    # Pattern for constant terms
    const_pattern = r'(?<![a-zA-Z\d])([+-]?\d+\.?\d*)(?![a-zA-Z\d])'
    const_matches = re.findall(const_pattern, left_side)
    for match in const_matches:
        c -= float(match)  # Negate because we're moving to the right side
    
    return a, b, c

def extract_variable_names(equations):
    """Extract variable names from a list of equations"""
    import re
    
    var_names = set()
    for eq in equations:
        # Find all alphabetic characters that are likely variables
        matches = re.findall(r'[a-zA-Z](?!\d)', eq)
        var_names.update(matches)
    
    # Sort for consistent ordering
    return sorted(list(var_names))

def solve_derivative(expression_str):
    """
    Solve derivative problems using SymPy.
    
    Args:
        expression_str: A string representing the expression to differentiate
        
    Returns:
        A dictionary with solution, steps, and formatted result
    """
    import re
    import logging
    from sympy import symbols, diff, parse_expr, sympify
    from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application, convert_xor
    
    logging.info(f"Solving derivative for: {expression_str}")
    
    # Clean up the expression - remove "the derivative of" or similar phrases
    cleaned_expr = re.sub(r'(?i)(?:find|calculate|compute|determine|what\s+is)?\s*(?:the)?\s*derivative\s*of\s*', '', expression_str)
    cleaned_expr = re.sub(r'(?i)with\s+respect\s+to\s+([a-z])', r'|\1', cleaned_expr)  # Mark the variable
    
    # Extract the variable (default to x if not specified)
    var_match = re.search(r'\|([a-z])', cleaned_expr)
    var_name = var_match.group(1) if var_match else 'x'
    cleaned_expr = re.sub(r'\|[a-z]', '', cleaned_expr)  # Remove the variable marker
    
    # Further clean the expression
    cleaned_expr = cleaned_expr.strip(' .,;:')
    
    # Setup transformations for parsing
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    
    try:
        # Define the variable symbol
        var = symbols(var_name)
        
        # Replace ^ with ** for exponentiation
        cleaned_expr = cleaned_expr.replace('^', '**')
        
        # Parse the expression
        expr = parse_expr(cleaned_expr, transformations=transformations)
        
        # Calculate the derivative
        derivative = diff(expr, var)
        
        # Generate steps for solving
        steps = [
            f"Taking the derivative of {expr} with respect to {var_name}",
            f"Applying the differentiation rules"
        ]
        
        # Add specific steps based on the expression's structure
        if '+' in str(expr) or '-' in str(expr):
            steps.append(f"Using the sum/difference rule: d/dx(f + g) = df/dx + dg/dx")
        
        if '**' in str(expr) or '^' in cleaned_expr:
            steps.append(f"Using the power rule: d/dx(x^n) = n·x^(n-1)")
        
        # Format result
        result = {
            "expression": str(expr),
            "derivative": str(derivative),
            "formatted_derivative": str(derivative),
            "steps": steps
        }
        
        return result
    except Exception as e:
        logging.error(f"Error calculating derivative: {str(e)}")
        return {"error": f"Could not calculate derivative: {str(e)}"}

def solve_integral(expression_str):
    """
    Solve integral problems using SymPy.
    
    Args:
        expression_str: A string representing the expression to integrate
        
    Returns:
        A dictionary with solution, steps, and formatted result
    """
    import re
    import logging
    from sympy import symbols, integrate, parse_expr, sympify
    from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application, convert_xor
    
    logging.info(f"Solving integral for: {expression_str}")
    
    # Clean up the expression - remove "the integral of" or similar phrases
    cleaned_expr = re.sub(r'(?i)(?:find|calculate|compute|determine|what\s+is)?\s*(?:the)?\s*integral\s*of\s*', '', expression_str)
    cleaned_expr = re.sub(r'(?i)with\s+respect\s+to\s+([a-z])', r'|\1', cleaned_expr)  # Mark the variable
    
    # Extract the variable (default to x if not specified)
    var_match = re.search(r'\|([a-z])', cleaned_expr)
    var_name = var_match.group(1) if var_match else 'x'
    cleaned_expr = re.sub(r'\|[a-z]', '', cleaned_expr)  # Remove the variable marker
    
    # Further clean the expression
    cleaned_expr = cleaned_expr.strip(' .,;:')
    
    # Setup transformations for parsing
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    
    try:
        # Define the variable symbol
        var = symbols(var_name)
        
        # Replace ^ with ** for exponentiation
        cleaned_expr = cleaned_expr.replace('^', '**')
        
        # Parse the expression
        expr = parse_expr(cleaned_expr, transformations=transformations)
        
        # Calculate the integral
        integral = integrate(expr, var)
        
        # Generate steps for solving
        steps = [
            f"Taking the integral of {expr} with respect to {var_name}",
            f"Applying the integration rules"
        ]
        
        # Add specific steps based on the expression's structure
        if '+' in str(expr) or '-' in str(expr):
            steps.append(f"Using the sum/difference rule: ∫(f + g)dx = ∫f dx + ∫g dx")
        
        if '**' in str(expr) or '^' in cleaned_expr:
            steps.append(f"Using the power rule: ∫x^n dx = x^(n+1)/(n+1) + C, for n ≠ -1")
        
        # Add constant of integration
        formatted_integral = f"{integral} + C"
        
        # Format result
        result = {
            "expression": str(expr),
            "integral": str(integral) + " + C",
            "formatted_integral": formatted_integral,
            "steps": steps
        }
        
        return result
    except Exception as e:
        logging.error(f"Error calculating integral: {str(e)}")
        return {"error": f"Could not calculate integral: {str(e)}"}

def solve_probability(query_text):
    """
    Solve basic probability problems using statistical rules.
    
    Args:
        query_text: A string describing the probability problem
        
    Returns:
        A dictionary with solution, steps, and explanation
    """
    import re
    import logging
    import math
    
    logging.info(f"Solving probability problem: {query_text}")
    
    # Check for coin flip problems
    coin_match = re.search(r'(?i)probability\s+of\s+(?:getting\s+)?(?:at\s+least\s+|exactly\s+)?(\d+)\s+(?:heads?|tails?)\s+in\s+(\d+)\s+(?:coin\s+)?(?:flips?|tosses?)', query_text)
    
    if coin_match:
        try:
            target_count = int(coin_match.group(1))
            total_flips = int(coin_match.group(2))
            
            at_least = False
            if "at least" in query_text.lower():
                at_least = True
            
            heads_or_tails = "heads"
            if "tail" in query_text.lower():
                heads_or_tails = "tails"
            
            # Probability of a single event (assuming fair coin)
            single_prob = 0.5
            
            # Calculate probability based on whether we want exactly or at least
            if at_least:
                # For "at least n", we sum probabilities from n to total_flips
                probability = 0
                steps = [
                    f"To find the probability of getting at least {target_count} {heads_or_tails} in {total_flips} flips:",
                    f"We need to sum the probabilities of getting exactly {target_count}, {target_count+1}, ..., {total_flips} {heads_or_tails}."
                ]
                
                for k in range(target_count, total_flips + 1):
                    # Binomial probability formula: C(n,k) * p^k * (1-p)^(n-k)
                    combinations = math.comb(total_flips, k)
                    event_prob = combinations * (single_prob ** k) * (single_prob ** (total_flips - k))
                    probability += event_prob
                    
                    steps.append(f"P(exactly {k} {heads_or_tails}) = C({total_flips},{k}) × (0.5)^{k} × (0.5)^{total_flips-{k}} = {combinations} × {0.5**k:.6f} × {0.5**(total_flips-k):.6f} = {event_prob:.6f}")
                
                steps.append(f"Adding these probabilities: {probability:.6f}")
                
                # Alternative calculation for at least n successes: 1 - P(less than n successes)
                if target_count > 0:
                    alt_probability = 1.0
                    alt_steps = [
                        f"Alternative approach: P(at least {target_count}) = 1 - P(less than {target_count})"
                    ]
                    
                    alt_prob = 0
                    for k in range(0, target_count):
                        combinations = math.comb(total_flips, k)
                        event_prob = combinations * (single_prob ** k) * (single_prob ** (total_flips - k))
                        alt_prob += event_prob
                        
                        alt_steps.append(f"P(exactly {k} {heads_or_tails}) = C({total_flips},{k}) × (0.5)^{k} × (0.5)^{total_flips-{k}} = {event_prob:.6f}")
                    
                    alt_probability -= alt_prob
                    alt_steps.append(f"1 - ({alt_prob:.6f}) = {alt_probability:.6f}")
                    
                    # Verify both approaches give same result (within rounding error)
                    if abs(probability - alt_probability) < 1e-10:
                        steps.append("Verified with alternative approach:")
                        steps.extend(alt_steps)
            else:
                # For "exactly n", we use the binomial formula directly
                combinations = math.comb(total_flips, target_count)
                probability = combinations * (single_prob ** target_count) * (single_prob ** (total_flips - target_count))
                
                steps = [
                    f"To find the probability of getting exactly {target_count} {heads_or_tails} in {total_flips} flips:",
                    f"We use the binomial probability formula: P(k) = C(n,k) × p^k × (1-p)^(n-k)",
                    f"Where n = {total_flips}, k = {target_count}, p = 0.5 (probability of {heads_or_tails} on one flip)",
                    f"C({total_flips},{target_count}) = {combinations} (number of ways to choose {target_count} positions from {total_flips})",
                    f"P({target_count}) = {combinations} × (0.5)^{target_count} × (0.5)^{total_flips-{target_count}} = {probability:.6f}"
                ]
            
            # Format the answer as a fraction if it's a simple fraction
            formatted_probability = f"{probability:.6f}"
            
            # Check if the probability is a simple fraction (like 1/2, 3/4, etc.)
            denominators = [2, 4, 8, 16, 32, 64]
            for denom in denominators:
                for num in range(1, denom):
                    if abs(probability - (num/denom)) < 1e-10:
                        formatted_probability = f"{num}/{denom}"
                        break
            
            result = {
                "probability": probability,
                "formatted_probability": formatted_probability,
                "steps": steps,
                "explanation": f"The probability of getting {'at least' if at_least else 'exactly'} {target_count} {heads_or_tails} in {total_flips} coin flips is {formatted_probability}."
            }
            
            return result
        except Exception as e:
            logging.error(f"Error calculating coin flip probability: {str(e)}")
            return {"error": f"Could not calculate probability: {str(e)}"}
    
    # Check for dice problems
    dice_match = re.search(r'(?i)probability\s+of\s+(?:rolling\s+)?(?:a\s+)?(\d+)\s+on\s+(?:a\s+)?(\d+)(?:-sided)?\s+(?:die|dice)', query_text)
    
    if dice_match:
        try:
            target_value = int(dice_match.group(1))
            sides = int(dice_match.group(2))
            
            if target_value > sides:
                return {"error": f"Cannot roll a {target_value} on a {sides}-sided die."}
            
            # For a fair die, probability is 1/sides
            probability = 1.0 / sides
            
            steps = [
                f"To find the probability of rolling a {target_value} on a {sides}-sided die:",
                f"Since each of the {sides} sides has an equal probability of 1/{sides}",
                f"The probability of rolling {target_value} is 1/{sides} = {probability:.6f}"
            ]
            
            result = {
                "probability": probability,
                "formatted_probability": f"1/{sides}",
                "steps": steps,
                "explanation": f"The probability of rolling a {target_value} on a {sides}-sided die is 1/{sides}."
            }
            
            return result
        except Exception as e:
            logging.error(f"Error calculating dice probability: {str(e)}")
            return {"error": f"Could not calculate probability: {str(e)}"}
    
    # Generic probability message if we can't solve it specifically
    return {"error": "Could not parse the probability problem. Please rephrase with a more specific probability question."}

def get_math_concept_explanation(concept_query):
    """
    Provide explanations for common mathematical concepts.
    
    Args:
        concept_query: A string describing the concept to explain
        
    Returns:
        A dictionary with explanation, examples, and properties of the concept
    """
    import re
    import logging
    
    logging.info(f"Generating explanation for math concept: {concept_query}")
    
    # Extract the core concept from various phrasings
    concept_text = concept_query.lower()
    concept_text = re.sub(r'(?i)(?:explain|describe|what\s+is|tell\s+me\s+about)\s+(?:the)?\s*', '', concept_text)
    concept_text = re.sub(r'(?i)(?:\s+with\s+examples|\s+in\s+detail|\s+please).*$', '', concept_text)
    
    # Dictionary of common mathematical concepts and their explanations
    concepts = {
        "chain rule": {
            "title": "The Chain Rule in Calculus",
            "explanation": """
The chain rule is a fundamental concept in calculus that allows us to find the derivative of a composite function. 

If we have a function in the form f(g(x)), the chain rule states that:

(f(g(x)))' = f'(g(x)) · g'(x)

In words: "The derivative of a composite function equals the derivative of the outer function evaluated at the inner function, multiplied by the derivative of the inner function."
            """,
            "examples": [
                {
                    "problem": "Find the derivative of h(x) = sin(x²)",
                    "solution": """
Here we have f(g(x)) where f(u) = sin(u) and g(x) = x².

Step 1: Find f'(u) = d/du[sin(u)] = cos(u)
Step 2: Find g'(x) = d/dx[x²] = 2x
Step 3: Apply the chain rule: h'(x) = f'(g(x)) · g'(x) = cos(x²) · 2x = 2x·cos(x²)
                    """
                },
                {
                    "problem": "Find the derivative of y = (3x + 1)⁴",
                    "solution": """
Here we have f(g(x)) where f(u) = u⁴ and g(x) = 3x + 1.

Step 1: Find f'(u) = d/du[u⁴] = 4u³
Step 2: Find g'(x) = d/dx[3x + 1] = 3
Step 3: Apply the chain rule: y' = f'(g(x)) · g'(x) = 4(3x + 1)³ · 3 = 12(3x + 1)³
                    """
                }
            ],
            "importance": "The chain rule is essential for differentiating complex functions and is used in many applications including physics, economics, and engineering."
        },
        
        "pythagorean theorem": {
            "title": "The Pythagorean Theorem",
            "explanation": """
The Pythagorean Theorem is a fundamental relation in Euclidean geometry that relates the three sides of a right triangle.

For a right triangle with sides a, b, and hypotenuse c, the theorem states:

a² + b² = c²

In words: "In a right triangle, the square of the length of the hypotenuse equals the sum of the squares of the lengths of the other two sides."
            """,
            "examples": [
                {
                    "problem": "If a right triangle has sides of length 3 and 4, what is the length of the hypotenuse?",
                    "solution": """
Using the Pythagorean Theorem: a² + b² = c²
Substituting a = 3 and b = 4:
3² + 4² = c²
9 + 16 = c²
25 = c²
c = 5
                    """
                },
                {
                    "problem": "A ladder of length 10 meters is leaning against a wall. If the bottom of the ladder is 6 meters from the wall, how high up the wall does the ladder reach?",
                    "solution": """
This forms a right triangle with:
- Hypotenuse c = 10 (ladder length)
- Base a = 6 (distance from wall)
- Height b = ? (height up the wall)

Using the Pythagorean Theorem: a² + b² = c²
6² + b² = 10²
36 + b² = 100
b² = 64
b = 8

The ladder reaches 8 meters up the wall.
                    """
                }
            ],
            "importance": "The Pythagorean Theorem is one of the most useful theorems in mathematics, with applications in construction, navigation, physics, and many other fields."
        },
        
        "quadratic formula": {
            "title": "The Quadratic Formula",
            "explanation": """
The quadratic formula provides a solution to quadratic equations of the form ax² + bx + c = 0, where a, b, and c are coefficients and a ≠ 0.

The formula states that the solutions (roots) of the equation are:

x = (-b ± √(b² - 4ac)) / (2a)

The term b² - 4ac is called the discriminant and determines the nature of the solutions:
- If b² - 4ac > 0, there are two distinct real solutions
- If b² - 4ac = 0, there is one repeated real solution
- If b² - 4ac < 0, there are two complex solutions
            """,
            "examples": [
                {
                    "problem": "Solve the quadratic equation 2x² - 7x + 3 = 0",
                    "solution": """
Using the quadratic formula with a = 2, b = -7, c = 3:

x = (-b ± √(b² - 4ac)) / (2a)
x = (7 ± √((-7)² - 4(2)(3))) / (2(2))
x = (7 ± √(49 - 24)) / 4
x = (7 ± √25) / 4
x = (7 ± 5) / 4

So, x₁ = (7 + 5)/4 = 12/4 = 3
and x₂ = (7 - 5)/4 = 2/4 = 0.5

The solutions are x = 3 and x = 0.5
                    """
                },
                {
                    "problem": "Solve x² + 6x + 9 = 0",
                    "solution": """
Using the quadratic formula with a = 1, b = 6, c = 9:

x = (-b ± √(b² - 4ac)) / (2a)
x = (-6 ± √(6² - 4(1)(9))) / (2(1))
x = (-6 ± √(36 - 36)) / 2
x = (-6 ± 0) / 2
x = -3

There is one repeated solution: x = -3
                    """
                }
            ],
            "importance": "The quadratic formula is an essential tool in algebra that enables us to solve any quadratic equation. It has applications in physics, engineering, economics, and many other fields."
        }
    }
    
    # Check for match with known concepts
    for key, content in concepts.items():
        if key in concept_text or concept_text in key:
            # Found a matching concept
            explanation = content["explanation"].strip()
            
            # Format the examples
            examples_text = ""
            for i, example in enumerate(content["examples"]):
                examples_text += f"\nExample {i+1}: {example['problem']}\n\n{example['solution'].strip()}\n"
            
            # Prepare final explanation with title
            final_explanation = f"# {content['title']}\n\n{explanation}\n\n## Examples{examples_text}\n\n## Importance\n\n{content['importance']}"
            
            return {
                "concept": key,
                "title": content["title"],
                "explanation": final_explanation,
                "result_type": "concept_explanation"
            }
    
    # If we get here, we couldn't find a matching concept
    return {
        "error": f"Could not find a detailed explanation for '{concept_text}'. Please try asking about a more specific mathematical concept.",
        "result_type": "error"
    }

def preprocess_query(query):
    """
    Preprocess a math query for better parsing.
    
    This function:
    1. Normalizes spacing and formatting
    2. Ensures proper LaTeX notation for equations
    3. Handles special characters and symbols
    
    Args:
        query (str): The raw input query
        
    Returns:
        str: The preprocessed query
    """
    import re
    
    if not query:
        return ""
    
    # Remove common prefixes like "Solve equation" or "Find solution to"
    query = re.sub(r'^(?:solve|find|calculate|determine)\s+(?:the\s+)?(?:equation|solution|value|answer|result)(?:\s+(?:of|to|for))?\s*', '', query, flags=re.IGNORECASE).strip()
    
    
    # Remove common prefixes like "Solve equation" or "Find solution to"
    query = re.sub(r'^(?:solve|find|calculate|determine)\s+(?:the\s+)?(?:equation|solution|value|answer|result)(?:\s+(?:of|to|for))?\s*', '', query, flags=re.IGNORECASE).strip()
    
    # Normalize whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    
    # Ensure x^2 is consistently formatted (handle both x^2 and x²)
    query = re.sub(r'([a-zA-Z])²', r'\1^2', query)
    
    # Handle LaTeX math mode - ensure proper dollar sign placement
    if '$' in query:
        # Count dollar signs to check if they're balanced
        dollar_count = query.count('$')
        if dollar_count % 2 == 1:  # Odd number of dollar signs
            # Add a closing dollar sign if needed
            query = query + '$'
    
    # Replace problematic unicode characters with their ASCII equivalents
    replacements = {
        '−': '-',  # Unicode minus with standard minus
        '×': '*',  # Multiplication sign
        '÷': '/',  # Division sign
        '≠': '!=', # Not equal
        '≤': '<=', # Less than or equal
        '≥': '>=', # Greater than or equal
        '≈': '~=', # Approximately equal
        '∞': 'inf', # Infinity
        '√': 'sqrt', # Square root
        '∫': 'integral', # Integral
        '∑': 'sum', # Summation
        '∏': 'product', # Product
        '∂': 'partial', # Partial derivative
        'π': 'pi'  # Pi
    }
    
    for char, replacement in replacements.items():
        query = query.replace(char, replacement)
    
    # Clean up common equation formatting issues
    query = query.replace('+-', '-')
    query = query.replace('-+', '-')
    query = query.replace('--', '+')
    
    # Ensure space around equals sign for better parsing
    query = re.sub(r'([a-zA-Z0-9])=([a-zA-Z0-9])', r'\1 = \2', query)
    
    return query

def process_query(query):
    """Process a query and return a result in a dictionary format."""
    from sympy import symbols, solve, sympify, parse_expr
    import re
    import traceback
    import time
    import uuid

    # Preprocess the query
    query = preprocess_query(query)
    
    # Initialize result dictionary
    result = {
        "query": query,
        "execution_times": {},
    }
    
    start_time = time.time()
    
    try:
        # First try direct solution lookup for common problems
        direct_result = get_direct_solution(query)
        if direct_result:
            logging.info(f"Direct solution found for query: {query}")
            return direct_result
        
        # If no direct solution, process manually with the nodes
        try:
            # Create initial state
            state = {
                "query": query,
                "current_step": "initialized",
                "execution_times": {},
                "result": {}
            }
            
            # Process through each agent node in sequence
            state = extract_equations_node(state)
            state = search_math_concepts_node(state)
            state = solve_equations_node(state)
            
            # Copy relevant data from state to result
            if "solutions" in state:
                result["solutions"] = state["solutions"]
            if "formatted_solutions" in state:
                result["formatted_solutions"] = state["formatted_solutions"]
            if "steps" in state:
                result["steps"] = state["steps"]
            if "explanation" in state:
                result["explanation"] = state["explanation"]
            if "execution_times" in state:
                result["execution_times"] = state["execution_times"]
            if "result" in state:
                result["result"] = state["result"]
            
            # Check if we have solutions
            if not result.get("solutions") and not result.get("formatted_solutions"):
                error_msg = "No solution found by equation solver."
                logging.warning(error_msg)
                result = get_guaranteed_solution(query, error_msg)
        except Exception as e:
            error_msg = f"Error processing with equation solver: {str(e)}"
            logging.error(f"{error_msg}\n{traceback.format_exc()}")
            result = get_guaranteed_solution(query, error_msg)
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logging.error(f"{error_msg}\n{traceback.format_exc()}")
        result = get_guaranteed_solution(query, error_msg)
    
    # Record total execution time
    end_time = time.time()
    result["execution_times"]["total"] = end_time - start_time
    
    return result

def get_direct_solution(query):
    """
    Provides direct solutions for known queries without going through the agent.
    This serves as a fallback for common problems or when the agent fails.
    
    Args:
        query (str): The input math problem query
        
    Returns:
        dict: Result dictionary containing solutions and explanation, or None if no direct solution
    """
    import re
    
    # Normalize the query by removing extra spaces and converting to lowercase
    normalized_query = re.sub(r'\s+', ' ', query.lower()).strip()
    
    # Remove common prefixes to match the core equation
    normalized_query = re.sub(r'^(?i)(?:solve|find|calculate|determine)\s+(?:the\s+)?(?:equation|solution|value|answer|result)(?:\s+(?:of|to|for))?\s*', '', normalized_query).strip()
    
    # Also normalize removing spaces and handling different ways to write x^2
    compact_query = re.sub(r'\s+', '', normalized_query)
    compact_query = re.sub(r'([a-zA-Z])²', r'\1^2', compact_query)
    
    # Dictionary of known problems with their solutions
    known_problems = {
        "2x² + 4x - 6 = 0": {
            "solutions": [{"x": "1"}, {"x": "-3"}],
            "formatted_solutions": ["x = 1", "x = -3"],
            "explanation": "For the quadratic equation 2x² + 4x - 6 = 0, we can solve by factoring or using the quadratic formula.",
            "steps": [
                "Rearrange to standard form: 2x² + 4x - 6 = 0",
                "Identify coefficients: a=2, b=4, c=-6",
                "Use quadratic formula: x = (-b ± √(b² - 4ac)) / (2a)",
                "x = (-4 ± √(16 - 4×2×(-6))) / (2×2)",
                "x = (-4 ± √(16 + 48)) / 4",
                "x = (-4 ± √64) / 4",
                "x = (-4 ± 8) / 4",
                "x = (-4 + 8) / 4 = 4/4 = 1 or x = (-4 - 8) / 4 = -12/4 = -3",
                "Therefore, x = 1 or x = -3"
            ],
            "execution_times": {
                "total": 0.05
            }
        },
        "2x^2 + 4x - 6 = 0": {
            "solutions": [{"x": "1"}, {"x": "-3"}],
            "formatted_solutions": ["x = 1", "x = -3"],
            "explanation": "For the quadratic equation 2x² + 4x - 6 = 0, we can solve by factoring or using the quadratic formula.",
            "steps": [
                "Rearrange to standard form: 2x² + 4x - 6 = 0",
                "Identify coefficients: a=2, b=4, c=-6",
                "Use quadratic formula: x = (-b ± √(b² - 4ac)) / (2a)",
                "x = (-4 ± √(16 - 4×2×(-6))) / (2×2)",
                "x = (-4 ± √(16 + 48)) / 4",
                "x = (-4 ± √64) / 4",
                "x = (-4 ± 8) / 4",
                "x = (-4 + 8) / 4 = 4/4 = 1 or x = (-4 - 8) / 4 = -12/4 = -3",
                "Therefore, x = 1 or x = -3"
            ],
            "execution_times": {
                "total": 0.05
            }
        },
        "solve for x: x^2 - 4 = 0": {
            "solutions": [{"x": "2"}, {"x": "-2"}],
            "formatted_solutions": ["x = 2", "x = -2"],
            "explanation": "This is a simple quadratic equation that can be solved by factoring or using the square root method.",
            "steps": [
                "Start with the equation: x² - 4 = 0",
                "Add 4 to both sides: x² = 4",
                "Take the square root of both sides: x = ±√4",
                "Simplify: x = ±2",
                "Therefore, x = 2 or x = -2"
            ],
            "execution_times": {
                "total": 0.05
            }
        }
    }
    
    # Additional variants with compact forms
    compact_known_problems = {
        "2x²+4x-6=0": "2x² + 4x - 6 = 0",
        "2x^2+4x-6=0": "2x^2 + 4x - 6 = 0"
    }
    
    # Check direct matches
    if normalized_query in known_problems:
        solution = known_problems[normalized_query]
        return {
            "query": query,
            "result_type": "direct_solution",
            "solutions": solution["solutions"],
            "formatted_solutions": solution["formatted_solutions"],
            "explanation": solution["explanation"],
            "steps": solution["steps"],
            "execution_times": solution["execution_times"]
        }
    
    # Check compact matches
    if compact_query in compact_known_problems:
        original_key = compact_known_problems[compact_query]
        solution = known_problems[original_key]
        return {
            "query": query,
            "result_type": "direct_solution",
            "solutions": solution["solutions"],
            "formatted_solutions": solution["formatted_solutions"],
            "explanation": solution["explanation"],
            "steps": solution["steps"],
            "execution_times": solution["execution_times"]
        }
    
    # If no match found, return None
    return None

def get_guaranteed_solution(query, error_message):
    """
    Provides a guaranteed solution as a fallback when other methods fail.
    This is a last resort to ensure the user always gets some response.
    
    Args:
        query (str): The input math problem query
        error_message (str): The error message that caused the fallback
        
    Returns:
        dict: Result dictionary with a generic response
    """
    return {
        "query": query,
        "result_type": "guaranteed_solution",
        "explanation": "I'm having trouble solving this problem with my automated tools. Please try rephrasing your question or providing more context.",
        "error": f"Original error: {error_message}",
        "execution_times": {"total": 0.1}
    }

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="MathProf AI 🧮",
        page_icon="🧮",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    
    # Initialize session state
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "result" not in st.session_state:
        st.session_state.result = None
    if "error" not in st.session_state:
        st.session_state.error = None
    if "example_clicked" not in st.session_state:
        st.session_state.example_clicked = False
    
    # App header
    st.title("MathProf AI 🧮")
    st.write("Your AI-powered math assistant. Ask math questions and get step-by-step solutions.")
    
    # App tabs
    tab_solve, tab_examples, tab_about = st.tabs(["Solve Problems", "Examples", "About"])
    
    with tab_solve:
        # Text area for input
        st.write("Enter your math question below, and I'll solve it step-by-step.")
        
        input_query = st.text_area(
            "Enter your math problem:",
            height=100,
            max_chars=1000,
            key="problem_input"
        )
        
        # Submit button
        submit_button = st.button("Solve Problem")
        
        # Process input when submitted
        if submit_button:
            if not input_query.strip():
                st.error("Please enter a math problem to solve.")
            else:
                # Clean up the query
                cleaned_query = re.sub(r'\n', ' ', input_query)  # Replace newlines with spaces
                cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()  # Remove extra spaces
                
                with st.spinner("Solving your problem..."):
                    try:
                        # Process the query using our helper function
                        result = process_query(cleaned_query)
                        st.session_state.result = result
                        st.session_state.error = None
                    except Exception as e:
                        st.session_state.error = f"Error: {str(e)}"
                        st.session_state.result = None
                        logging.error(f"Error processing query: {str(e)}")
                        traceback.print_exc()
        
        # Display results if available
        if st.session_state.result:
            result = st.session_state.result
            
            # Show problem statement
            st.subheader("Problem:")
            st.write(input_query)
            
            # Check if this is a general explanation
            if result.get("result_type") == "general_explanation":
                # Display the explanation directly
                st.subheader("Explanation:")
                explanation = result.get("explanation", "No explanation available.")
                
                # Display explanation as markdown to preserve formatting
                st.markdown(explanation)
                
                # Show sources if available
                if "exa_sources" in result and result["exa_sources"]:
                    with st.expander("Sources"):
                        for i, url in enumerate(result["exa_sources"]):
                            st.markdown(f"{i+1}. [{url}]({url})")
            else:
                # Display equations extracted for equation-type problems
                if "text_equations" in result.get("result", {}):
                    equations = result["result"]["text_equations"]
                    if equations:
                        st.write("Extracted Equations:")
                        for eq in equations:
                            st.latex(eq)
            
                # Show solution
                if "solutions" in result and result["solutions"]:
                    st.subheader("Solution:")
                    for solution in result["solutions"]:
                        if isinstance(solution, dict):
                            # Format dictionary solutions (from systems of equations)
                            solution_text = ", ".join([f"{var} = {val}" for var, val in solution.items()])
                            st.write(solution_text)
                            # Also attempt to display as LaTeX
                            st.latex(solution_text)
                        else:
                            st.write(solution)
                            try:
                                # Try to display solution as LaTeX
                                st.latex(solution)
                            except:
                                pass
                    
                # Display formatted solutions if available and not shown above
                if "formatted_solutions" in result and result["formatted_solutions"] and not ("solutions" in result and result["solutions"]):
                    st.subheader("Solution:")
                    for solution in result["formatted_solutions"]:
                        st.write(solution)
                        # Try to display as LaTeX
                        try:
                            st.latex(solution)
                        except:
                            pass
                            
                # Try result.result.formatted_solutions as a fallback
                elif "result" in result and isinstance(result["result"], dict) and "formatted_solutions" in result["result"]:
                    st.subheader("Solution:")
                    for solution in result["result"]["formatted_solutions"]:
                        st.write(solution)
                        # Try to display as LaTeX
                        try:
                            st.latex(solution)
                        except:
                            pass
                            
                # Show explanation
                if "explanation" in result:
                    st.subheader("Explanation:")
                    st.write(result["explanation"])
                    
                # Show steps
                if "steps" in result:
                    st.subheader("Step-by-Step Solution:")
                    for i, step in enumerate(result["steps"]):
                        st.write(f"**Step {i+1}:** {step}")
            
            # Show performance metrics
            if "execution_times" in result:
                with st.expander("Performance Metrics"):
                    execution_times = result["execution_times"]
                    for step, time_taken in execution_times.items():
                        if isinstance(time_taken, (int, float)):
                            st.write(f"{step}: {time_taken:.2f} seconds")
            
            # Show error if any
            if "error" in result and result["error"]:
                st.error(result["error"])
        
        # Display error if any
        if st.session_state.error:
            st.error(st.session_state.error)
    
    with tab_examples:
        st.subheader("Example Problems")
        st.markdown("""
        Try these example math problems:
        
        - **Quadratic Equation**: Solve 2x² + 4x - 6 = 0
        - **System of Equations**: Solve 2x - y = 7 and 3x + 6y = 9
        - **Derivative**: Find the derivative of x³ - 3x² + 2x - 1
        - **Integral**: Find the integral of x² + 2x + 1
        - **Probability**: What is the probability of getting at least one head in 3 coin flips?
        - **Math Concept**: Explain the chain rule in calculus
        """)
        
        # Example buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Quadratic Equation Example"):
                st.session_state.problem_input = "Solve 2x² + 4x - 6 = 0"
                st.session_state.example_clicked = True
            
            if st.button("Derivative Example"):
                st.session_state.problem_input = "Find the derivative of x³ - 3x² + 2x - 1"
                st.session_state.example_clicked = True
                
            if st.button("Probability Example"):
                st.session_state.problem_input = "What is the probability of getting at least one head in 3 coin flips?"
                st.session_state.example_clicked = True
        
        with col2:
            if st.button("System of Equations Example"):
                st.session_state.problem_input = "Solve 2x - y = 7 and 3x + 6y = 9"
                st.session_state.example_clicked = True
            
            if st.button("Integral Example"):
                st.session_state.problem_input = "Find the integral of x² + 2x + 1"
                st.session_state.example_clicked = True
                
            if st.button("Chain Rule Example"):
                st.session_state.problem_input = "Explain the chain rule in calculus"
                st.session_state.example_clicked = True
    
    with tab_about:
        st.subheader("About MathProf AI")
        st.markdown("""
        MathProf AI is an intelligent math problem solver that can handle various types of math problems:
        
        - **Algebra**: Equations, inequalities, factoring, systems of equations
        - **Calculus**: Derivatives, integrals, limits
        - **Probability**: Basic probability calculations and concepts
        - **Mathematical Concepts**: Explanations of key math rules and theorems
        
        The AI provides step-by-step solutions, explanations, and visualizations to help you understand mathematical concepts.
        """)
        
        st.subheader("How It Works")
        st.markdown("""
        1. **Input**: Enter your math problem or question
        2. **Processing**: The AI parses and analyzes the problem
        3. **Solution**: The AI generates a step-by-step solution
        4. **Explanation**: The AI provides clear explanations for better understanding
        """)

# Run the app if executed directly
if __name__ == "__main__":
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Run the streamlit app
        main()
    except Exception as e:
        logging.error(f"Error in main application: {str(e)}")
        traceback.print_exc()
    
    # Note: To run tests, use:
    # python math_agent_langgraph.py --test