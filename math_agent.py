#!/usr/bin/env python3
"""
Math Agent - An AI-powered assistant for solving math problems
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

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Time operations for performance tracking
def time_operation(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
openai_max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "4000"))

client = OpenAI(api_key=openai_api_key)

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
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Function to search for relevant math knowledge
def search_math_knowledge(query, limit=5):
    if not qdrant_client:
        return []
    
    try:
        query_embedding = get_embedding(query)
        search_result = qdrant_client.search(
            collection_name=qdrant_collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
        results = []
        for result in search_result:
            results.append({
                "payload": result.payload,
                "score": result.score
            })
        
        return results
    except Exception as e:
        logger.error(f"Error searching math knowledge: {e}")
        return []

def extract_equations(text):
    """
    Extract mathematical equations from text, supporting LaTeX.
    
    Args:
        text (str): Input text containing math equations
        
    Returns:
        list: Extracted equations
    """
    equations = []
    
    # LaTeX pattern with $ delimiters
    latex_pattern = r'\$(.*?)\$'
    latex_matches = re.findall(latex_pattern, text)
    equations.extend(latex_matches)
    
    # Extract equations using double $ delimiters (display math)
    display_math_pattern = r'\$\$(.*?)\$\$'
    display_matches = re.findall(display_math_pattern, text)
    equations.extend(display_matches)
    
    # Extract equations in LaTeX \begin{equation} environment
    equation_env_pattern = r'\\begin\{equation\}(.*?)\\end\{equation\}'
    equation_matches = re.findall(equation_env_pattern, text, re.DOTALL)
    equations.extend(equation_matches)
    
    # Also try to extract plain text equations like "2x + 3 = 7"
    plain_eq_pattern = r'(\d+[\+\-\*/x\^ ]+\d+\s*=\s*\d+)'
    plain_matches = re.findall(plain_eq_pattern, text)
    equations.extend(plain_matches)
    
    return equations

def parse_equation(equation_str):
    """
    Parse a mathematical equation string into SymPy expression.
    
    Args:
        equation_str (str): String representation of the equation
        
    Returns:
        sympy.Expr: SymPy expression object
    """
    try:
        # First try to parse as LaTeX
        try:
            expr = parse_latex(equation_str)
            return expr
        except Exception:
            pass
        
        # If LaTeX parsing fails, try as a regular expression
        try:
            # Handle basic equations like "x + 3 = 7"
            if "=" in equation_str:
                left_side, right_side = equation_str.split("=", 1)
                left_expr = parse_expr(left_side.strip())
                right_expr = parse_expr(right_side.strip())
                return Eq(left_expr, right_expr)
            else:
                # Handle expressions without equals sign
                return parse_expr(equation_str.strip())
        except Exception:
            pass
        
        # If both parsing methods fail, return None
        return None
    except Exception as e:
        logger.error(f"Error parsing equation '{equation_str}': {e}")
        return None

def solve_equation(equation):
    """
    Solve a mathematical equation using SymPy.
    
    Args:
        equation (sympy.Expr or str): The equation to solve
        
    Returns:
        list: Solution(s) to the equation
    """
    try:
        if isinstance(equation, str):
            equation = parse_equation(equation)
        
        if equation is None:
            return None
        
        # Handle different types of equations
        if isinstance(equation, Eq):
            # Algebraic equation
            var = list(equation.free_symbols)
            if len(var) == 0:
                return []
            var = var[0]  # Assume first symbol is the variable to solve for
            return solve(equation, var)
        
        # If it's a differential equation
        if equation.has(sympy.Derivative):
            # Get variables and functions
            funcs = list(equation.atoms(sympy.Function))
            if len(funcs) == 0:
                return None
            return dsolve(equation, funcs[0])
        
        # Default to treating it as a regular expression to solve
        vars = list(equation.free_symbols)
        if len(vars) == 0:
            return equation
        return solve(equation, vars[0])
    
    except Exception as e:
        logger.error(f"Error solving equation: {e}")
        return None

def calculate_derivative(expression_str):
    """
    Calculate the derivative of a mathematical expression.
    
    Args:
        expression_str (str): The expression to differentiate
        
    Returns:
        sympy.Expr: The derivative of the expression
    """
    try:
        expression = parse_equation(expression_str)
        if expression is None:
            return None
        
        # If expression is an equation, differentiate both sides
        if isinstance(expression, Eq):
            left_deriv = diff(expression.lhs, x)
            right_deriv = diff(expression.rhs, x)
            return Eq(left_deriv, right_deriv)
        
        # Otherwise, differentiate the expression
        return diff(expression, x)
    
    except Exception as e:
        logger.error(f"Error calculating derivative: {e}")
        return None

def evaluate_integral(expression_str):
    """
    Calculate the indefinite integral of a mathematical expression.
    
    Args:
        expression_str (str): The expression to integrate
        
    Returns:
        sympy.Expr: The integral of the expression
    """
    try:
        expression = parse_equation(expression_str)
        if expression is None:
            return None
        
        return integrate(expression, x)
    
    except Exception as e:
        logger.error(f"Error evaluating integral: {e}")
        return None

def format_math_answer(answer):
    """
    Format a mathematical answer for display.
    
    Args:
        answer: The solution to format
        
    Returns:
        str: Formatted answer string
    """
    if answer is None:
        return "Could not solve the equation."
    
    if isinstance(answer, list):
        if not answer:
            return "No solutions found."
        return "Solutions: " + ", ".join([str(sol) for sol in answer])
    
    return str(answer)

def output_guardrails(answer):
    """
    Apply guardrails to the output to protect user privacy and ensure appropriate content.
    
    Args:
        answer (str): The original answer from the model
        
    Returns:
        str: The sanitized answer
    """
    try:
        # Store original mathematical expressions to restore later
        math_expressions = []
        
        def preserve_math(match):
            math_expressions.append(match.group(0))
            return f"__MATH_EXPR_{len(math_expressions)-1}__"
        
        # Replace math terms with placeholders
        math_pattern = r"(\$\$.*?\$\$)|(\$.*?\$)|(\\begin\{align\}.*?\\end\{align\})|(\\begin\{aligned\}.*?\\end\{aligned\})"
        answer_with_placeholders = re.sub(math_pattern, preserve_math, answer, flags=re.DOTALL)
        
        # PII detection patterns with fixed formatting
        pii_patterns = [
            (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]"),  # SSN
            (r"\b\d{3}-\d{3}-\d{4}\b", "[PHONE_REDACTED]"),  # Phone numbers
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_REDACTED]"),  # Email
            (r"\b[A-Z][a-z]{2,} [A-Z][a-z]{2,}\b", "[NAME_REDACTED]")  # Names (First Last format)
        ]
        
        # Apply PII detection
        for pattern, replacement in pii_patterns:
            answer_with_placeholders = re.sub(pattern, replacement, answer_with_placeholders)
        
        # Restore math terms
        for i, term in enumerate(math_expressions):
            answer_with_placeholders = answer_with_placeholders.replace(f"__MATH_EXPR_{i}__", term)
        
        return answer_with_placeholders
    except Exception as e:
        logger.warning(f"Error in output guardrails: {e}")
        return answer  # Return original answer if error occurs

def process_query(query, history=None):
    """
    Process a mathematical query and generate an answer.
    
    Args:
        query (str): User's mathematical question
        history (list): Conversation history
        
    Returns:
        str: Answer to the mathematical question
    """
    start_time = time.time()
    
    # Extract equations from the query
    equations = extract_equations(query)
    direct_solution = None
    
    # Try to solve equations directly
    if equations:
        for eq_str in equations:
            # Try to solve the equation
            solution = solve_equation(eq_str)
            if solution is not None:
                direct_solution = format_math_answer(solution)
                break
            
            # Try to calculate derivative if query contains keywords
            if "derivative" in query.lower() or "differentiate" in query.lower():
                derivative = calculate_derivative(eq_str)
                if derivative is not None:
                    direct_solution = f"The derivative is: ${derivative}$"
                    break
            
            # Try to evaluate integral if query contains keywords
            if "integral" in query.lower() or "integrate" in query.lower():
                integral = evaluate_integral(eq_str)
                if integral is not None:
                    direct_solution = f"The integral is: ${integral}$"
                    break
    
    # Retrieve relevant math knowledge
    math_knowledge = search_math_knowledge(query)
    knowledge_context = ""
    
    if math_knowledge:
        knowledge_context = "Relevant mathematical knowledge:\n"
        for i, item in enumerate(math_knowledge):
            content = item["payload"].get("content", "")
            knowledge_context += f"{i+1}. {content}\n"
    
    # If we have a direct solution, return it
    if direct_solution and "solve" in query.lower():
        answer = f"I solved this equation directly:\n\n{direct_solution}"
        
        # Time tracking
        if track_performance:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"Query processed in {execution_time:.4f} seconds")
        
        return output_guardrails(answer)
    
    # If we don't have a direct solution or it's a complex query, use OpenAI
    messages = [
        {"role": "system", "content": f"""You are a helpful math assistant. Answer the user's math question, showing step-by-step reasoning.
Remember to use proper LaTeX formatting for math expressions, like $x^2$ or $$\\int f(x) dx$$.

{knowledge_context}"""}
    ]
    
    # Add history if provided
    if history:
        for h in history:
            messages.append({"role": "user" if h["role"] == "user" else "assistant", "content": h["content"]})
    
    # Add the current query
    messages.append({"role": "user", "content": query})
    
    # If we have a direct solution, include it as a hint to the model
    if direct_solution:
        messages.append({"role": "system", "content": f"I calculated that the solution is: {direct_solution}"})
    
    # Generate response using OpenAI
    try:
        response = client.chat.completions.create(
            model=openai_model,
            messages=messages,
            temperature=openai_temperature,
            max_tokens=openai_max_tokens
        )
        
        answer = response.choices[0].message.content
        
        # Time tracking
        if track_performance:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"Query processed in {execution_time:.4f} seconds")
        
        return output_guardrails(answer)
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        
        # If OpenAI fails but we have a direct solution, return that
        if direct_solution:
            return output_guardrails(f"I solved this equation directly:\n\n{direct_solution}")
        
        return "I'm sorry, I encountered an error while processing your question. Please try again."

def main():
    st.title("Math Agent")
    st.write("Ask me any math question!")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if query := st.chat_input("Your question:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Process the query
        with st.spinner("Thinking..."):
            answer = process_query(query, st.session_state.messages)
        
        # Display assistant message
        with st.chat_message("assistant"):
            st.markdown(answer)
        
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main() 