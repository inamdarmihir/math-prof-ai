#!/usr/bin/env python3
"""
Enhanced Math Agent - An improved AI-powered assistant for solving math problems
with better model support and optimized prompts
"""

import os
import re
import logging
import time
import json
import sympy
from sympy import symbols, solve, Eq, diff, integrate, Matrix, limit, oo, simplify, expand, factor
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from sympy.solvers.ode import dsolve
from sympy.abc import x, y, z, t
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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
            logging.FileHandler(os.path.join(logs_dir, 'math_agent_enhanced.log')),
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
    logging.warning(f"Unable to setup file logging: {str(e)}. Using console logging only.")

# Create a logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
# Get other configuration from environment
openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
openai_temperature = float(os.getenv("MODEL_TEMPERATURE", "0.0"))
openai_max_tokens = int(os.getenv("MODEL_MAX_TOKENS", "4000"))

# Initialize OpenAI client
if not openai_api_key:
    logger.error("OpenAI API key not found in environment variables")
    client = None
else:
    try:
        client = OpenAI(api_key=openai_api_key)
        logger.info(f"OpenAI client initialized successfully with model {openai_model}")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        client = None

# Initialize Qdrant client for vector search
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_collection_name = os.getenv("QDRANT_COLLECTION", "math_knowledge")

if qdrant_url and qdrant_api_key:
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    logger.info(f"Connected to Qdrant at {qdrant_url}")
else:
    qdrant_client = None
    logger.warning("Qdrant connection not configured. Vector search will be disabled.")

# Enable/disable debug mode
debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
# Enable/disable performance tracking
track_performance = os.getenv("TRACK_PERFORMANCE", "false").lower() == "true"

# Function to get embeddings from OpenAI
def get_embedding(text):
    """Get embeddings for a text string using OpenAI's embedding API."""
    try:
        # Use text-embedding-ada-002 which produces 1536-dimensional vectors
        # This is what the current database is configured for
        response = client.embeddings.create(
            model="text-embedding-ada-002",  # Changed from text-embedding-3-large to match database expectation
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        return None

# Function to search for relevant knowledge in vector database
def search_vector_db(query_text, limit=5):
    """
    Search for relevant mathematical knowledge in vector database.
    
    Args:
        query_text: The search query
        limit: Maximum number of results to return
        
    Returns:
        List of relevant mathematical knowledge entries
    """
    if not qdrant_client:
        logger.warning("Vector search not available: Qdrant client not initialized")
        return []
        
    try:
        # Get query embedding
        query_vector = get_embedding(query_text)
        if not query_vector:
            return []
            
        # Search Qdrant collection
        search_result = qdrant_client.search(
            collection_name=qdrant_collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        # Extract and return knowledge entries
        knowledge_entries = []
        for result in search_result:
            knowledge_entries.append({
                "content": result.payload.get("content", ""),
                "category": result.payload.get("category", ""),
                "score": result.score
            })
            
        return knowledge_entries
    except Exception as e:
        logger.error(f"Error searching vector database: {str(e)}")
        return []

# Enhanced system prompt specifically designed for mathematical problem-solving
SYSTEM_PROMPT = """
You are an advanced mathematical problem-solving assistant with expertise in various mathematical domains
including algebra, calculus, geometry, trigonometry, probability, and more. Your goal is to provide
clear, accurate, and step-by-step solutions to mathematical problems.

Follow these guidelines when solving problems:

1. DOMAIN IDENTIFICATION:
   - Identify the mathematical domain of the problem (algebra, calculus, geometry, etc.)
   - Recognize specific subtopics (integration, differentiation, quadratic equations, etc.)

2. SOLUTION APPROACH:
   - Decide on the most appropriate method or formula to solve the problem
   - Break down complex problems into manageable steps
   - If multiple approaches exist, choose the most elegant or efficient one

3. STEP-BY-STEP SOLUTION:
   - Provide a clear, sequential solution with explicit reasoning for each step
   - Use proper mathematical notation with LaTeX formatting
   - Explain key concepts and transformations
   - Verify your answer by checking or providing a quick validation

4. COMMON PROBLEMS AND STRATEGIES:
   - For CALCULUS problems: Apply differentiation/integration rules carefully, use substitution methods when appropriate
   - For ALGEBRA problems: Properly factor expressions, carefully manipulate equations, use appropriate algebraic identities
   - For GEOMETRY problems: Draw diagrams mentally, apply geometric formulas correctly, use coordinate transformations when helpful
   - For PROBABILITY problems: Identify the correct probability distribution, calculate expectation and variance properly
   - For VECTOR problems: Decompose vectors into components, apply vector operations systematically
   - For MATRICES problems: Apply matrix operations correctly, understand determinants and eigenvalues

5. OUTPUT FORMATTING:
   - Use LaTeX for mathematical expressions: $x^2$ for inline math, \\begin{align}...\\end{align} for equations
   - Ensure proper formatting of fractions, exponents, integrals, and other mathematical notation
   - Present final answers clearly and in simplified form when possible

Always verify your final answer using substitution, dimensional analysis, or other validation methods when possible.
"""

# Function to process mathematical queries
def process_query(user_query, history=None):
    """
    Process a mathematical query by:
    1. Extracting equations
    2. Determining the mathematical domain
    3. Searching for relevant knowledge
    4. Structuring the query based on the domain
    5. Getting a response from OpenAI
    6. Tracking performance
    """
    start_time = time.time()
    try:
        logger.info(f"Processing query: {user_query}")
        
        # Extract equations from the query
        equations = extract_equations(user_query)
        if equations and equations != [user_query]:
            logger.info(f"Extracted equation(s): {', '.join(equations)}")
            
        # Search for relevant knowledge
        logger.info("Searching for relevant knowledge...")
        relevant_knowledge = []
        try:
            relevant_knowledge = search_vector_db(user_query)
            logger.info(f"Found {len(relevant_knowledge)} relevant knowledge entries")
        except Exception as e:
            logger.error(f"Error searching for knowledge: {str(e)}")
            # Continue without relevant knowledge
        
        # Determine the mathematical domain
        domain = determine_math_domain(user_query)
        logger.info(f"Detected math domain: {domain}")
        
        # Structure the query based on the domain
        structured_query = structure_query_by_domain(user_query, domain, relevant_knowledge)
        
        # Override default model settings with GPT-4
        model = os.environ.get("OPENAI_MODEL", "gpt-4-turbo-preview")
        temperature = float(os.environ.get("MODEL_TEMPERATURE", "0.0"))
        logger.info(f"Sending request to {model} with temperature {temperature}")
        
        # Get the response from OpenAI
        response = client.chat.completions.create(
            model=model,
            messages=structured_query,
            temperature=temperature,
            max_tokens=int(os.environ.get("MODEL_MAX_TOKENS", "4000"))
        )
        
        # Format the response
        answer_text = response.choices[0].message.content
        formatted_response = format_math_response(answer_text)
        
        # Track performance
        processing_time = time.time() - start_time
        logger.info(f"Query processed in {processing_time:.2f} seconds")
        
        if os.environ.get("TRACK_PERFORMANCE", "true").lower() == "true":
            track_query_performance(user_query, domain, processing_time)
            
        # Extract confidence from the response
        confidence = 0.0
        confidence_pattern = r'Confidence:?\s*(\d+(?:\.\d+)?)%?'
        confidence_match = re.search(confidence_pattern, formatted_response)
        if confidence_match:
            confidence_str = confidence_match.group(1)
            try:
                confidence = float(confidence_str)
                # Convert percentage to decimal if needed
                if confidence > 1.0:
                    confidence /= 100.0
            except ValueError:
                pass
                
        return {
            "answer": formatted_response,
            "confidence": confidence,
            "domain": domain,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        processing_time = time.time() - start_time
        return {
            "answer": f"I encountered an error while processing your query: {str(e)}",
            "confidence": 0.0,
            "domain": "error",
            "processing_time": processing_time
        }

def determine_math_domain(query):
    """
    Determine the mathematical domain of a query using keyword analysis.
    
    Args:
        query: The user's query
        
    Returns:
        The detected mathematical domain
    """
    query_lower = query.lower()
    
    # Domain keywords
    domain_keywords = {
        "calculus": ["derivative", "integral", "differentiate", "integrate", "limit", "converge", "series"],
        "algebra": ["solve", "equation", "simplify", "factor", "expand", "polynomial", "quadratic"],
        "geometry": ["circle", "triangle", "square", "rectangle", "area", "volume", "perimeter"],
        "trigonometry": ["sin", "cos", "tan", "angle", "radian", "degree", "trigonometric"],
        "probability": ["probability", "random", "expected", "variance", "distribution", "sample"],
        "vectors": ["vector", "dot product", "cross product", "magnitude", "direction"],
        "matrices": ["matrix", "determinant", "eigenvalue", "eigenvector", "inverse"],
        "complex_numbers": ["complex", "imaginary", "real part", "imaginary part", "modulus", "argument"],
        "differential_equations": ["differential equation", "ode", "pde", "solution"]
    }
    
    # Check for domain keywords
    domain_scores = {}
    for domain, keywords in domain_keywords.items():
        score = sum(1 for keyword in keywords if keyword in query_lower)
        domain_scores[domain] = score
    
    # Get domain with highest score
    max_score = max(domain_scores.values())
    if max_score > 0:
        max_domains = [domain for domain, score in domain_scores.items() if score == max_score]
        return max_domains[0]
    
    # Default to general if no specific domain detected
    return "general"

def structure_query_by_domain(query, domain, relevant_knowledge=None):
    """
    Structure the query with domain-specific prompting strategies.
    
    Args:
        query: The user's query
        domain: The detected mathematical domain
        relevant_knowledge: Optional relevant knowledge from vector database
        
    Returns:
        Structured query for the language model
    """
    # Base prompt structure
    structured_query = f"""
MATHEMATICAL PROBLEM:
{query}

DOMAIN: {domain.upper()}

"""
    
    # Add domain-specific instructions
    if domain == "calculus":
        structured_query += """
APPROACH:
1. Identify what type of calculus problem this is (derivative, integral, limit, etc.)
2. Apply the appropriate calculus rules and techniques
3. Show each step of the calculation clearly
4. Verify the result if possible
"""
    elif domain == "algebra":
        structured_query += """
APPROACH:
1. Identify the algebraic structure and type of problem
2. Apply appropriate algebraic manipulations and transformations
3. Show all steps in solving the equation or simplifying the expression
4. Check your solution by substitution if applicable
"""
    elif domain == "geometry":
        structured_query += """
APPROACH:
1. Visualize the geometric objects and their relationships
2. Identify relevant formulas and theorems
3. Apply geometric reasoning step by step
4. Calculate the final result with proper units if applicable
"""
    elif domain == "probability":
        structured_query += """
APPROACH:
1. Identify the probability space and events
2. Determine the appropriate probability principles to apply
3. Calculate probabilities step by step
4. Verify that the results are consistent with probability axioms
"""
    
    # Add relevant knowledge if available
    if relevant_knowledge:
        structured_query += "\nRELEVANT KNOWLEDGE:\n"
        for i, entry in enumerate(relevant_knowledge[:3]):
            structured_query += f"{i+1}. {entry['content']}\n"
    
    # Add final instructions
    structured_query += """
INSTRUCTIONS:
- Provide a clear, step-by-step solution
- Use proper mathematical notation with LaTeX formatting
- Explain your reasoning at each step
- Verify your final answer
- Present the final answer clearly
"""
    
    return structured_query

def extract_equations(text):
    """Extract mathematical equations from text."""
    try:
        # More comprehensive regex pattern to capture full equations
        # This pattern looks for equations with =, <, >, ≤, ≥ symbols
        equation_pattern = r'([^.;]*(=|<|>|≤|≥)[^.;]*)'
        equations = re.findall(equation_pattern, text)
        
        # If the above pattern doesn't find anything, look for expressions
        if not equations:
            # Look for mathematical expressions like "2x^2 - 5x + 3"
            expression_pattern = r'([0-9]+[a-zA-Z]?(?:\^[0-9]+)?(?:[\+\-\*\/][0-9]+[a-zA-Z]?(?:\^[0-9]+)?)+)'
            expressions = re.findall(expression_pattern, text)
            if expressions:
                return expressions
            
            # Check for specific problem formats (like "Solve: 2x^2 - 5x + 3 = 0")
            solve_pattern = r'(?:Solve|Find|Evaluate)[:\s]+([^.;]+)'
            solve_matches = re.findall(solve_pattern, text)
            if solve_matches:
                return solve_matches
        
        # Process the captured equations to get just the equation part
        result_equations = []
        for eq, _ in equations:  # (full match, operator)
            eq = eq.strip()
            if eq:
                result_equations.append(eq)
                
        # If we still don't have any equations, just return the full text
        # as a fallback - let the LLM handle it
        if not result_equations and "solve" in text.lower():
            # Get text after "solve" keyword
            solve_parts = text.lower().split("solve")
            if len(solve_parts) > 1:
                return [solve_parts[1].strip()]
                
        return result_equations if result_equations else [text]
    except Exception as e:
        logger.warning(f"Error extracting equations: {str(e)}")
        return [text]  # Return full text if extraction fails

def format_math_response(text):
    """
    Format and sanitize math response, improving LaTeX formatting.
    
    Args:
        text: The text to format
        
    Returns:
        Formatted text with improved LaTeX
    """
    # Fix common LaTeX formatting issues
    
    # Remove square brackets often added by LLMs
    text = re.sub(r'\[\s*\\begin\{align\}', r'\\begin{align}', text)
    text = re.sub(r'\\end\{align\}\s*\]', r'\\end{align}', text)
    
    # Ensure proper LaTeX escaping
    text = re.sub(r'\\\\', r'\\\\\\\\', text)  # Properly escape backslashes
    
    # Fix align environments
    text = re.sub(r'\\begin\{align\}\s*\\begin\{align\*\}', r'\\begin{align*}', text)
    text = re.sub(r'\\end\{align\*\}\s*\\end\{align\}', r'\\end{align*}', text)
    
    # Fix equation spacing
    text = re.sub(r'([^\\])([=><])([^=><])', r'\1 \2 \3', text)
    
    # Fix fraction formatting
    text = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'\\frac{\1}{\2}', text)
    
    # Clean up other formatting
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Remove excess newlines
    
    return text

def track_query_performance(query, domain, processing_time):
    """
    Track performance metrics for queries.
    
    Args:
        query: The user's query
        domain: The mathematical domain
        processing_time: The processing time in seconds
    """
    timestamp = datetime.now().isoformat()
    metrics = {
        "timestamp": timestamp,
        "query": query,
        "domain": domain,
        "processing_time": processing_time,
        "model": openai_model
    }
    
    # Write metrics to performance log
    try:
        with open(os.path.join(logs_dir, 'performance_metrics.jsonl'), 'a+') as f:
            f.write(json.dumps(metrics) + '\n')
    except Exception as e:
        logger.error(f"Error writing performance metrics: {str(e)}")

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Enhanced Math Agent",
        page_icon="➗",
        layout="wide"
    )
    
    st.title("Enhanced Math Agent 🧮")
    st.subheader("Solve mathematical problems with step-by-step solutions")
    
    # Sidebar for settings
    st.sidebar.title("Settings")
    if openai_model:
        st.sidebar.success(f"Using model: {openai_model}")
    else:
        st.sidebar.error("OpenAI API key not configured")
    
    # Input area
    user_query = st.text_area("Enter your mathematical problem:", height=100)
    
    if st.button("Solve"):
        if not user_query:
            st.warning("Please enter a mathematical problem to solve.")
        elif not client:
            st.error("Cannot process query: OpenAI client not initialized")
        else:
            with st.spinner("Solving..."):
                answer = process_query(user_query)
                st.markdown(answer["answer"])

if __name__ == "__main__":
    main() 