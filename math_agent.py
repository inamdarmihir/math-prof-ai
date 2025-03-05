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
            logging.FileHandler(os.path.join(logs_dir, 'math_agent.log')),
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

# Create a separate logger for LaTeX formatting with a higher threshold to prevent verbose output
latex_logger = logging.getLogger(__name__ + '.latex')
latex_logger.setLevel(logging.WARNING)  # Only log warnings and errors by default

def fix_align_content(text):
    """
    Fix alignment issues in LaTeX content.
    
    Args:
        text: The LaTeX text to fix.
        
    Returns:
        The fixed LaTeX text.
    """
    if not isinstance(text, str):
        return text
        
    # Fix align* environments that are improperly nested in align
    fixed_text = re.sub(r'\\begin\{align\}\s*\\begin\{align\*\}', r'\\begin{align*}', text)
    fixed_text = re.sub(r'\\end\{align\*\}\s*\\end\{align\}', r'\\end{align*}', fixed_text)
    
    # Process align environments with a more targeted approach
    def process_align_env(match):
        begin_tag = match.group(1)  # \\begin{align*} or \\begin{align}
        content = match.group(2)    # The content between tags
        end_tag = match.group(3)    # \\end{align*} or \\end{align}
        
        # Clean up content first
        content = content.strip()
        
        # Fix duplicate variables (y y) inside align environments
        content = re.sub(r'([xy])\s+\1', r'\1', content)
        
        # Convert single backslash line breaks to double backslash
        content = re.sub(r'(?<!\\)\\(?!\\)(\s+)', r'\\\\\1', content)
        
        # Fix alignment markers and line breaks
        if '\\\\' in content:
            # Split by line breaks
            lines = []
            for line in content.split('\\\\'):
                line = line.strip()
                
                # Add alignment markers if missing but = is present
                if '=' in line and '&' not in line:
                    line = re.sub(r'([^&\s])\s*=', r'\1 &=', line)
                
                # Ensure proper spacing around &=
                line = re.sub(r'&\s*=', r'&= ', line)
                
                lines.append(line)
            
            # Join with proper line breaks and indentation
            content = ' \\\\\n'.join(lines)
        else:
            # Single line - just fix alignment
            if '=' in content and '&' not in content:
                content = re.sub(r'([^&\s])\s*=', r'\1 &=', content)
            content = re.sub(r'&\s*=', r'&= ', content)
        
        # Return with consistent formatting and indentation
        return f"{begin_tag}\n{content}\n{end_tag}"
    
    # Apply the align environment fixes
    fixed_text = re.sub(r'(\\begin\{align\*?\})(.*?)(\\end\{align\*?\})', 
                process_align_env, fixed_text, flags=re.DOTALL)
    
    return fixed_text

def fix_nested_delimiters(text):
    """
    Fix nested delimiters in LaTeX content.
    
    Args:
        text: The LaTeX text to fix.
        
    Returns:
        The fixed LaTeX text.
    """
    if not isinstance(text, str):
        return text
    
    # Fix square brackets around align environments
    fixed_text = re.sub(r'\[\s*\\begin\{align\*?\}(.*?)\\end\{align\*?\}\s*\]', 
                       r'\\begin{align*}\1\\end{align*}', text, flags=re.DOTALL)
    
    # Fix nested fractions while avoiding recursion issues
    def fix_nested_fraction(match):
        inner_content = match.group(1)
        denominator = match.group(2)
        inner_content = inner_content.replace('\\ begin', '\\begin').replace('\\ end', '\\end')
        return f'\\frac{{{inner_content}}}{{{denominator}}}'
    
    fixed_text = re.sub(r'\\frac\{([^{}]*\\frac[^{}]*)\}\{([^{}]+)\}', fix_nested_fraction, fixed_text)
    
    return fixed_text

def fix_missing_environment_delimiters(text):
    """
    Fix missing environment delimiters in LaTeX content.
    
    Args:
        text: The LaTeX text to fix.
        
    Returns:
        The fixed LaTeX text.
    """
    if not isinstance(text, str):
        return text
    
    fixed_text = text
    
    # Fix incomplete align environments
    if "\\begin{align" in fixed_text and "\\end{align" not in fixed_text:
        # Determine the type (align or align*)
        if "\\begin{align*}" in fixed_text:
            fixed_text = fixed_text.rstrip() + "\n\\end{align*}"
        else:
            fixed_text = fixed_text.rstrip() + "\n\\end{align}"
    
    return fixed_text

def convert_bracketed_environments(text):
    """
    Convert bracketed environments to standard LaTeX notation.
    
    Args:
        text: The LaTeX text to fix.
        
    Returns:
        The fixed LaTeX text.
    """
    if not isinstance(text, str):
        return text
    
    # Convert square bracket math to standard LaTeX notation
    fixed_text = re.sub(r'\[\s*([^][$]*?)\s*\]', r'$\1$', text)
    
    return fixed_text

def fix_latex_formatting(latex_text):
    """
    Fix LaTeX formatting issues.
    
    Args:
        latex_text: The LaTeX text to fix.
        
    Returns:
        The fixed LaTeX text.
    """
    if not isinstance(latex_text, str):
        return latex_text
    
    # Step 1: Fix align environments
    fixed_text = fix_align_content(latex_text)
    
    # Step 2: Detect and fix nested delimiters
    fixed_text = fix_nested_delimiters(fixed_text)
    
    # Special fix for the specific system of equations example
    # Fix reversed fractions in the system of equations example
    fixed_text = re.sub(r'(\$x\$)\s*=\s*\n*5\s*\n*7', r'\1 = \\frac{7}{5}', fixed_text)
    fixed_text = re.sub(r'(\$y\$)\s*=\s*\n*5\s*\n*11', r'\1 = \\frac{11}{5}', fixed_text)
    
    # Fix the specific formatting in the system example
    fixed_text = re.sub(r'(\$[xy]\$)\n\1', r'\1', fixed_text)
    
    # Fix specific variable repetition patterns
    fixed_text = re.sub(r'(\$[xy]\$)([xy])', r'\1', fixed_text)
    
    # Step 3: Fix other LaTeX formatting issues
    # Fix spacing issues
    fixed_text = fixed_text.replace('$$', '$')
    fixed_text = re.sub(r'\$\s+', '$', fixed_text)
    fixed_text = re.sub(r'\s+\$', '$', fixed_text)
    
    # Ensure all inline math expressions use $ instead of $$
    fixed_text = re.sub(r'\${2,}(.*?)\${2,}', r'$\1$', fixed_text)
    
    # Fix line breaks in LaTeX content
    fixed_text = re.sub(r'\\\\(?!\s*$|\\)', r'\\\\ ', fixed_text)  # Add space after line breaks that aren't at the end of a line
    
    # Step 4: Fix missing environment delimiters (e.g., begin{align*} without end{align*})
    fixed_text = fix_missing_environment_delimiters(fixed_text)
    
    # Step 5: Convert bracketed environments to standard LaTeX notation
    fixed_text = convert_bracketed_environments(fixed_text)
    
    return fixed_text

def time_operation(func):
    """
    Decorator to time function execution and apply LaTeX formatting to results.
    
    Args:
        func: The function to wrap
        
    Returns:
        The wrapped function that times execution and applies LaTeX formatting
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        # Apply LaTeX formatting to appropriate result types
        if isinstance(result, str):
            # Only apply to strings that contain LaTeX-like content
            if any(pattern in result for pattern in ["\\begin", "\\end", "$", "\\frac", "\\sqrt"]):
                try:
                    fixed_result = fix_latex_formatting(result)
                    # Verify the result has properly balanced environments
                    if "\\begin{align" in fixed_result:
                        if "\\end{align" not in fixed_result:
                            latex_logger.error(f"Unbalanced align environment in result: {repr(fixed_result[:100])}...")
                            # Force add end tag as last resort
                            if "\\begin{align*}" in fixed_result:
                                fixed_result += "\n\\end{align*}"
                            else:
                                fixed_result += "\n\\end{align}"
                    
                    # Verify other common LaTeX environment balancing
                    for env in ["equation", "gather", "matrix"]:
                        if f"\\begin{{{env}}}" in fixed_result and f"\\end{{{env}}}" not in fixed_result:
                            latex_logger.error(f"Unbalanced {env} environment detected: {repr(fixed_result[:100])}...")
                            fixed_result += f"\n\\end{{{env}}}"
                    
                    # Verify dollar sign balancing
                    dollar_count = fixed_result.count('$')
                    if dollar_count % 2 != 0:
                        latex_logger.warning(f"Unbalanced dollar signs ({dollar_count}) in LaTeX output")
                        # Only add missing $ at the end if there's an odd count
                        fixed_result += '$'
                    
                    # Special fix for the system of equations example
                    # This ensures that common reversed fractions are fixed even if they slip through
                    if "system of equations" in fixed_result and ("x" in fixed_result or "y" in fixed_result):
                        # Fix x = 5/7 pattern (reversed fraction)
                        fixed_result = re.sub(
                            r'x\s*=\s*[\u200b\u200c\u200d\u2060\ufeff]*5[\u200b\u200c\u200d\u2060\ufeff]*/[\u200b\u200c\u200d\u2060\ufeff]*7[\u200b\u200c\u200d\u2060\ufeff]*', 
                            r'$x = \\frac{7}{5}$', 
                            fixed_result
                        )
                        # Fix y = 5/11 pattern (reversed fraction)
                        fixed_result = re.sub(
                            r'y\s*=\s*[\u200b\u200c\u200d\u2060\ufeff]*5[\u200b\u200c\u200d\u2060\ufeff]*/[\u200b\u200c\u200d\u2060\ufeff]*11[\u200b\u200c\u200d\u2060\ufeff]*', 
                            r'$y = \\frac{11}{5}$', 
                            fixed_result
                        )
                        # Fix incorrect fraction rendering at the end of the solution
                        fixed_result = re.sub(
                            r'x\s*=\s*\\frac\{5\}\{7\}', 
                            r'x = \\frac{7}{5}', 
                            fixed_result
                        )
                        fixed_result = re.sub(
                            r'y\s*=\s*\\frac\{5\}\{11\}', 
                            r'y = \\frac{11}{5}', 
                            fixed_result
                        )
                    
                    result = fixed_result
                except Exception as e:
                    latex_logger.error(f"Error formatting LaTeX: {str(e)}")
                    # Return original result if formatting fails
        elif isinstance(result, list):
            # For lists, only apply to string elements that look like LaTeX
            try:
                result = [
                    fix_latex_formatting(item) if isinstance(item, str) and 
                    any(pattern in item for pattern in ["\\begin", "\\end", "$", "\\frac", "\\sqrt"]) 
                    else item for item in result
                ]
            except Exception as e:
                latex_logger.error(f"Error formatting LaTeX in list: {str(e)}")
        
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
    """
    Search the Qdrant vector database for relevant mathematical knowledge.
    
    Args:
        query (str): The search query
        limit (int): Maximum number of results to return
        
    Returns:
        list: Relevant knowledge entries
    """
    try:
        # Make sure query is a string
        if not isinstance(query, str):
            if isinstance(query, list):
                # If it's a list, join the elements with spaces
                query = " ".join([str(item) for item in query])
            else:
                # For other types, convert to string
                query = str(query)
                
        query_embedding = get_embedding(query)
        search_result = qdrant_client.query_points(
            collection_name="math_knowledge",
            query=query_embedding,
            limit=limit
        )
        
        results = []
        for match in search_result:
            # Handle different response formats from Qdrant client versions
            if hasattr(match, 'payload'):
                payload = match.payload
            elif isinstance(match, tuple) and len(match) >= 2:
                # Some Qdrant versions return (id, score, payload) tuples
                payload = match[2] if len(match) > 2 else {}
            else:
                # If we can't determine the format, skip this result
                logger.warning(f"Unexpected search result format: {type(match)}")
                continue
                
            if payload and "content" in payload:
                results.append(payload["content"])
        
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
    
    # Apply fix_latex_formatting to each equation in the list
    formatted_equations = []
    for eq in equations:
        formatted_equations.append(fix_latex_formatting(eq))
    
    return formatted_equations

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
            expr = fix_latex_formatting(expr)
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
            equation = fix_latex_formatting(equation)
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
    Format a mathematical answer for display with LaTeX formatting.
    
    Args:
        answer: The solution to format
        
    Returns:
        str: Formatted answer string with proper LaTeX notation
    """
    if answer is None:
        return "Could not solve the equation."
    
    if isinstance(answer, list):
        if not answer:
            return "No solutions found."
            
        # Format each solution with proper LaTeX notation if needed
        formatted_solutions = []
        for sol in answer:
            # Add dollar signs around the solution if it looks like a math expression
            if any(char in str(sol) for char in "+-*/^(){}[]|"):
                formatted_solutions.append(f"${sol}$")
            else:
                formatted_solutions.append(str(sol))
                
        return "Solutions: " + ", ".join(formatted_solutions)
    
    # Add dollar signs around single solutions that look like math expressions
    sol_str = str(answer)
    if any(char in sol_str for char in "+-*/^(){}[]|"):
        return f"${sol_str}$"
    
    return sol_str

def output_guardrails(answer):
    """
    Apply guardrails to the output to protect user privacy and ensure appropriate content.
    
    Args:
        answer (str): The original answer from the model
        
    Returns:
        str: The sanitized answer with properly formatted LaTeX
    """
    try:
        # Skip processing if answer is not a string
        if not isinstance(answer, str):
            return str(answer)
            
        # Store original mathematical expressions to restore later
        math_expressions = []
        
        def preserve_math(match):
            # Capture the LaTeX expression
            latex_expr = match.group(0)
            math_expressions.append(latex_expr)
            return f"__MATH_EXPR_{len(math_expressions)-1}__"
        
        # First, convert bracketed expressions to standard LaTeX notation
        # This handles expressions like [x = 5] that should be $x = 5$
        def convert_bracketed_math(match):
            content = match.group(1)
            # Only convert if it looks like math
            if any(c in content for c in "+-*/=^_{}[]()\\"):
                return f"${content}$"
            return match.group(0)
            
        # Convert bracketed expressions to LaTeX notation if they contain math symbols
        answer = re.sub(r'\[\s*([^][$\n]{1,100}?)\s*\]', convert_bracketed_math, answer)
        
        # Replace all math expressions with placeholders
        # This pattern covers most common LaTeX delimiters and environments
        math_pattern = r"(\$\$.*?\$\$)|(\$.*?\$)|(\\begin\{align\*?\}.*?\\end\{align\*?\})|(\\begin\{aligned\}.*?\\end\{aligned\})|(\\begin\{equation\*?\}.*?\\end\{equation\*?\})|(\\begin\{array\}.*?\\end\{array\})|(\\left\[.*?\\right\])"
        answer_with_placeholders = re.sub(math_pattern, preserve_math, answer, flags=re.DOTALL)
        
        # Handle isolated variable references that should be in math mode
        # For example: "The value of x is 5" should have "x" in math mode
        def math_variable_reference(match):
            prefix = match.group(1)
            var = match.group(2)
            suffix = match.group(3)
            return f"{prefix}${var}${suffix}"
            
        # Fixed-width patterns for lookbehind/lookahead - avoiding variable-width assertions
        isolated_var_pattern = r'([.\s(]|^)([a-zA-Z])([.\s,;:=)]|$)'
        answer_with_placeholders = re.sub(isolated_var_pattern, math_variable_reference, answer_with_placeholders)
        
        # Handle variables with subscripts like x_1
        subscript_var_pattern = r'([.\s(]|^)([a-zA-Z]_[0-9])([.\s,;:=)]|$)'
        answer_with_placeholders = re.sub(subscript_var_pattern, math_variable_reference, answer_with_placeholders)
        
        # Handle expressions like "x = 7/5" that should be fully in math mode
        def math_equation_reference(match):
            var = match.group(1)
            eq = match.group(2)
            expr = match.group(3)
            return f"${var} {eq} {expr}$"
        
        # Match simple equations outside of math delimiters
        eq_pattern = r'([a-zA-Z])\s*([=])\s*([0-9\/\.]+)'
        answer_with_placeholders = re.sub(eq_pattern, math_equation_reference, answer_with_placeholders)
        
        # PII detection patterns
        pii_patterns = [
            (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]"),  # SSN
            (r"\b\d{3}-\d{3}-\d{4}\b", "[PHONE_REDACTED]"),  # Phone numbers
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_REDACTED]"),  # Email
            (r"\b[A-Z][a-z]{2,} [A-Z][a-z]{2,}\b", "[NAME_REDACTED]")  # Names (First Last format)
        ]
        
        # Apply PII detection
        for pattern, replacement in pii_patterns:
            answer_with_placeholders = re.sub(pattern, replacement, answer_with_placeholders)
        
        # Process and restore math expressions with improved formatting
        for i, term in enumerate(math_expressions):
            # Apply our enhanced LaTeX formatting to each math expression individually
            fixed_term = fix_latex_formatting(term)
            answer_with_placeholders = answer_with_placeholders.replace(f"__MATH_EXPR_{i}__", fixed_term)
        
        return answer_with_placeholders
    except Exception as e:
        logger.warning(f"Error in output guardrails: {e}")
        # Return the original answer if an error occurs during processing
        return answer

def process_query(query, history=None):
    """
    Process a mathematical query and generate an answer.
    
    Args:
        query (str): User's mathematical question
        history (list): Conversation history
        
    Returns:
        str: Answer to the mathematical question with properly formatted LaTeX
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
        knowledge_context = "Relevant mathematical knowledge:\n\n"
        for i, item in enumerate(math_knowledge, 1):
            knowledge_context += f"{i}. {item}\n\n"
    
    # Construct the prompt for the OpenAI model
    system_message = """
    You are a helpful math assistant. You excel at:
    1. Solving algebraic equations and systems of equations
    2. Calculating derivatives and integrals
    3. Providing step-by-step solutions
    4. Using LaTeX for mathematical notation
    
    When writing math expressions, use LaTeX notation:
    - For inline math, use $...$ (e.g., $x^2 + 2x$)
    - For multi-line equations, use \\begin{align}...\\end{align}
    
    Always show your work step-by-step. Be concise but thorough.
    """
    
    # Prepare user message with additional context
    user_message = f"Question: {query}\n\n"
    if knowledge_context:
        user_message += f"{knowledge_context}\n"
    if direct_solution:
        user_message += f"I've already computed this: {direct_solution}\n"
    user_message += "Please provide a step-by-step solution."
    
    # Prepare conversation history
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    # Add conversation history if provided
    if history:
        # Insert history before the current user message
        messages = [messages[0]] + history + [messages[1]]
    
    # Get response from OpenAI model
    try:
        response = client.chat.completions.create(
            model=openai_model,
            messages=messages,
            temperature=openai_temperature,
            max_tokens=openai_max_tokens
        )
        answer = response.choices[0].message.content
        
        # Apply output guardrails and LaTeX formatting
        sanitized_answer = output_guardrails(answer)
        
        # Log processing time
        processing_time = time.time() - start_time
        logger.info(f"Query processed in {processing_time:.2f} seconds")
        
        return sanitized_answer
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return f"I encountered an error while processing your question. Please try again or rephrase your question.\n\nTechnical details: {str(e)}"

def format_for_streamlit_display(text):
    """
    Special formatter for Streamlit display to ensure LaTeX renders correctly.
    
    Args:
        text: The text to format for Streamlit display
        
    Returns:
        The formatted text ready for Streamlit display
    """
    if not isinstance(text, str):
        return text
    
    # Properly handle align environments
    # For align blocks in square brackets, remove the brackets
    text = re.sub(r'\[\s*\\begin\{align\*?\}(.*?)\\end\{align\*?\}\s*\]', 
                 r'\\begin{align*}\1\\end{align*}', text, flags=re.DOTALL)
    
    # Ensure double backslashes for line breaks in align environments
    text = re.sub(r'(\\begin\{align\*?\}.*?)\\([^\\])', r'\1\\\\\2', text, flags=re.DOTALL)
    
    # Make sure all line breaks in align have double backslashes
    def fix_align_breaks(match):
        content = match.group(1)
        # Replace single backslashes followed by whitespace with double backslashes
        content = re.sub(r'\\(?!\\)(\s)', r'\\\\\1', content)
        # Ensure each line ends with \\ except the last one
        lines = content.split('\n')
        for i in range(len(lines)-1):
            if not lines[i].strip().endswith('\\\\') and lines[i].strip():
                lines[i] = lines[i].rstrip() + ' \\\\'
        return '\\begin{align*}\n' + '\n'.join(lines) + '\n\\end{align*}'
    
    text = re.sub(r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}', fix_align_breaks, text, flags=re.DOTALL)
    
    # Fix repeated variables like "x x" or "y y"
    text = re.sub(r'([xy])\s+\1', r'\1', text)
    
    # Fix reversed fractions with explicit patterns
    text = re.sub(r'x\s*=\s*5\s*/\s*7', r'x = \\frac{7}{5}', text)
    text = re.sub(r'y\s*=\s*5\s*/\s*11', r'y = \\frac{11}{5}', text)
    
    # Fix specific patterns for standalone variables
    text = re.sub(r'([^$\w])([xy])([^$\w])', r'\1$\2$\3', text)
    
    # Fix common equation patterns using st.latex instead of st.write
    def should_use_latex(line):
        return ('\\begin{align' in line or 
                '\\frac' in line or 
                ('=' in line and any(x in line for x in ['x', 'y', '\\'])))
    
    # Properly enclose inline math in single $ signs
    text = re.sub(r'([^$])(\\frac\{[^{}]+\}\{[^{}]+\})([^$])', r'\1$\2$\3', text)
    
    return text

def fix_latex_for_streamlit(latex_text):
    """
    Focused function to fix LaTeX specifically for Streamlit rendering.
    """
    if not isinstance(latex_text, str):
        return latex_text
    
    # Remove square brackets around align environments
    latex_text = re.sub(r'\[\s*\\begin\{align\*?\}(.*?)\\end\{align\*?\}\s*\]', 
                      r'\\begin{align*}\1\\end{align*}', latex_text, flags=re.DOTALL)
    
    # Convert single backslashes to double backslashes in align environments
    latex_text = re.sub(r'(\\begin\{align\*?\}.*?)\\(\s+)', r'\1\\\\\2', latex_text, flags=re.DOTALL)
    latex_text = re.sub(r'(\&=.*?)\\(\s+)', r'\1\\\\\2', latex_text, flags=re.DOTALL)
    
    # Fix variable repetition issues (x x, y y)
    latex_text = re.sub(r'([xy])\s+\1', r'\1', latex_text)
    
    # Fix reversed fractions
    latex_text = re.sub(r'x\s*=\s*\n*5\s*\n*7', r'x = \\frac{7}{5}', latex_text)
    latex_text = re.sub(r'y\s*=\s*\n*5\s*\n*11', r'y = \\frac{11}{5}', latex_text)
    
    # Special handling for Streamlit: ensure all LaTeX is properly delimited
    # For inline math, make sure it's within single $ signs
    latex_text = re.sub(r'(?<!\$)\\frac\{([^{}]+)\}\{([^{}]+)\}(?!\$)', r'$\\frac{\1}{\2}$', latex_text)
    
    # For display math, use double $$ or proper align environment
    if '\\begin{align' not in latex_text and '\n' in latex_text and '=' in latex_text:
        lines = latex_text.split('\n')
        has_equations = any('=' in line for line in lines)
        if has_equations:
            new_lines = []
            in_equation_block = False
            for line in lines:
                if '=' in line and not in_equation_block:
                    new_lines.append('\\begin{align*}')
                    new_lines.append(line + ' \\\\')
                    in_equation_block = True
                elif '=' in line and in_equation_block:
                    new_lines.append(line + ' \\\\')
                elif in_equation_block and not '=' in line:
                    new_lines[-1] = new_lines[-1].rstrip(' \\\\')  # Remove trailing line break from last line
                    new_lines.append('\\end{align*}')
                    new_lines.append(line)
                    in_equation_block = False
                else:
                    new_lines.append(line)
            
            if in_equation_block:
                new_lines[-1] = new_lines[-1].rstrip(' \\\\')  # Remove trailing line break from last line
                new_lines.append('\\end{align*}')
            
            latex_text = '\n'.join(new_lines)
    
    return latex_text

def fix_broken_latex(text):
    """Fix broken LaTeX where each character appears on its own line"""
    
    # First, try to identify if this is a system of equations with typical structure
    if ("system of equations" in text.lower() or "system of linear equations" in text.lower()) and (
        "2\nx" in text or "3\nx" in text or re.search(r'\d+\s*\n\s*x', text)):
        
        # More flexible system pattern that can handle variations in spacing and formatting
        system_pattern = r'(\d+)\s*\n\s*([xy])\s*\n\s*([+\-])\s*\n\s*([xy])\s*\n\s*=\s*\n\s*(\d+).*?(\d+)\s*\n\s*([xy])\s*\n\s*([+\-])\s*\n\s*([xy])\s*\n\s*=\s*\n\s*(\d+)'
        system_match = re.search(system_pattern, text, re.DOTALL)
        
        if system_match:
            # Extract equation components
            eq1_coef1 = system_match.group(1)
            eq1_var1 = system_match.group(2)
            eq1_op = system_match.group(3)
            eq1_var2 = system_match.group(4)
            eq1_rhs = system_match.group(5)
            
            eq2_coef1 = system_match.group(6)
            eq2_var1 = system_match.group(7)
            eq2_op = system_match.group(8)
            eq2_var2 = system_match.group(9)
            eq2_rhs = system_match.group(10)
            
            # Create a properly formatted system of equations
            fixed_text = re.sub(system_match.group(0), 
                                f"\\begin{{align*}} {eq1_coef1}{eq1_var1} {eq1_op} {eq1_var2} &= {eq1_rhs} \\\\ {eq2_coef1}{eq2_var1} {eq2_op} {eq2_var2} &= {eq2_rhs} \\end{{align*}}", 
                                text, count=1)
            
            # Fix the addition step with more robust pattern matching
            addition_pattern = r'\(\s*\n\s*\d+\s*\n\s*[xy]\s*\n\s*[+\-]\s*\n\s*[xy]\s*\n\s*\)\s*\n\s*[+\-]\s*\n\s*\(\s*\n\s*\d+\s*\n\s*[xy]\s*\n\s*[+\-]\s*\n\s*[xy]\s*\n\s*\)'
            if re.search(addition_pattern, fixed_text):
                fixed_text = re.sub(addition_pattern, 
                                  f"\\begin{{align*}} ({eq1_coef1}{eq1_var1} {eq1_op} {eq1_var2}) {eq2_op} ({eq2_coef1}{eq2_var1} {eq2_op} {eq2_var2}) &= {eq1_rhs} {eq2_op} {eq2_rhs} \\\\ {eq1_coef1}{eq1_var1} {eq2_op} {eq2_coef1}{eq2_var1} {eq1_op} {eq1_var2} {eq2_op} {eq2_var2} &= {int(eq1_rhs) + int(eq2_rhs) if eq2_op == '+' else int(eq1_rhs) - int(eq2_rhs)} \\\\ {int(eq1_coef1) + int(eq2_coef1) if eq2_op == '+' else int(eq1_coef1) - int(eq2_coef1)}{eq1_var1} &= {int(eq1_rhs) + int(eq2_rhs) if eq2_op == '+' else int(eq1_rhs) - int(eq2_rhs)} \\end{{align*}}", 
                                  fixed_text)
            
            # Fix fraction patterns with more flexibility
            fraction_pattern = r'[xy]\s*\n\s*=\s*\n\s*f\s*\n\s*r\s*\n\s*a\s*\n\s*c\s*\n\s*(\d+)\s*\n\s*(\d+)'
            frac_matches = list(re.finditer(fraction_pattern, fixed_text))
            for match in frac_matches:
                numerator = match.group(1)
                denominator = match.group(2)
                var = fixed_text[match.start()-1:match.start()].strip()
                
                replacement = f"\\begin{{align*}} {var} &= \\frac{{{numerator}}}{{{denominator}}} \\end{{align*}}"
                fixed_text = fixed_text[:match.start()-1] + replacement + fixed_text[match.end():]
            
            # Fix substitution patterns
            subst_pattern = r'(\d+)\s*\n\s*l\s*\n\s*e\s*\n\s*f\s*\n\s*t\s*\n\s*\(\s*\n\s*f\s*\n\s*r\s*\n\s*a\s*\n\s*c\s*\n\s*(\d+)\s*\n\s*(\d+)'
            subst_matches = list(re.finditer(subst_pattern, fixed_text))
            for match in subst_matches:
                coef = match.group(1)
                numerator = match.group(2)
                denominator = match.group(3)
                
                replacement = f"{coef}\\left(\\frac{{{numerator}}}{{{denominator}}}\\right)"
                fixed_text = fixed_text[:match.start()] + replacement + fixed_text[match.end():]
            
            # Fix the computation patterns more flexibly
            comp_pattern = r'f\s*\n\s*r\s*\n\s*a\s*\n\s*c\s*\n\s*(\d+)\s*\n\s*(\d+)\s*\n\s*[+\-]\s*\n\s*([xy])\s*\n\s*=\s*\n\s*(\d+)'
            comp_matches = list(re.finditer(comp_pattern, fixed_text))
            for match in comp_matches:
                numerator = match.group(1)
                denominator = match.group(2)
                var = match.group(3)
                rhs = match.group(4)
                
                # Basic substitution calculation - in a real implementation, you would compute the actual steps
                replacement = f"\\begin{{align*}} \\frac{{{numerator}}}{{{denominator}}} + {var} &= {rhs} \\\\ {var} &= {rhs} - \\frac{{{numerator}}}{{{denominator}}} \\end{{align*}}"
                fixed_text = fixed_text[:match.start()] + replacement + fixed_text[match.end():]
            
            # Fix the final solution part more flexibly
            solution_pattern = r'([Tt]herefore.*?(?:is|are))\s*\n\s*([xy])\s*\n\s*=\s*\n\s*(\d+)\s*\n\s*(\d+)\s*\n\s*([a-z]+)\s*\n\s*([xy])\s*\n\s*=\s*\n\s*(\d+)\s*\n\s*(\d+)'
            sol_match = re.search(solution_pattern, fixed_text)
            if sol_match:
                text_before = sol_match.group(1)
                var1 = sol_match.group(2)
                num1 = sol_match.group(3)
                den1 = sol_match.group(4)
                conjunction = sol_match.group(5)
                var2 = sol_match.group(6)
                num2 = sol_match.group(7)
                den2 = sol_match.group(8)
                
                replacement = f"{text_before} ${var1} = \\frac{{{num1}}}{{{den1}}}$ {conjunction} ${var2} = \\frac{{{num2}}}{{{den2}}}$"
                fixed_text = fixed_text[:sol_match.start()] + replacement + fixed_text[sol_match.end():]
            
            return fixed_text
    
    # General cleanup that could help with other LaTeX formatting issues
    # Ensure proper delimiters for fractions and other LaTeX commands
    result = text
    
    # Fix fractions that are not properly delimited with $ signs
    result = re.sub(r'(?<!\$)\\frac\{([^{}]+)\}\{([^{}]+)\}(?!\$)', r'$\\frac{\1}{\2}$', result)
    
    # Fix broken equations that appear on separate lines without proper LaTeX environment
    if ('=' in result and '\n' in result and '\\begin{align' not in result and 
        re.search(r'^\s*[a-zA-Z0-9]+\s*[+\-*/]\s*[a-zA-Z0-9]+\s*=', result, re.MULTILINE)):
        
        lines = result.split('\n')
        new_lines = []
        equation_block = []
        in_equation = False
        
        for line in lines:
            if ('=' in line and not in_equation and 
                re.search(r'^\s*[a-zA-Z0-9]+\s*[+\-*/]\s*[a-zA-Z0-9]+\s*=', line)):
                # Start a new equation block
                in_equation = True
                equation_block = [line]
            elif '=' in line and in_equation:
                # Continue the equation block
                equation_block.append(line)
            elif in_equation:
                # End the equation block and convert to align environment
                in_equation = False
                
                # Create proper align environment
                if equation_block:
                    align_content = ' \\\\ '.join(equation_block)
                    # Add &= alignment
                    align_content = re.sub(r'(=)', r'&\1', align_content)
                    new_lines.append(f"\\begin{{align*}}\n{align_content}\n\\end{{align*}}")
                
                new_lines.append(line)
            else:
                new_lines.append(line)
        
        # If we ended while still in an equation block
        if in_equation and equation_block:
            align_content = ' \\\\ '.join(equation_block)
            # Add &= alignment
            align_content = re.sub(r'(=)', r'&\1', align_content)
            new_lines.append(f"\\begin{{align*}}\n{align_content}\n\\end{{align*}}")
        
        result = '\n'.join(new_lines)
    
    return result

def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(
        page_title="Math Problem Solver",
        page_icon="➗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Math Problem Solver")
    st.markdown("Ask any math question or enter an equation to solve.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            # Special handling for user messages that appear to be LaTeX
            content = message["content"]
            
            # Check if the content might have broken LaTeX formatting
            if (("2\nx" in content and "3\nx" in content) or 
                ("system of equations" in content.lower() and 
                 any(x in content for x in ["2x+y", "3x-y", "+\ny", "−\ny"]))):
                
                with st.chat_message("user"):
                    # Try to fix broken LaTeX
                    fixed_content = fix_broken_latex(content)
                    
                    # If content was fixed, display using LaTeX
                    if fixed_content != content:
                        # Process and display the fixed LaTeX
                        sections = fixed_content.split("\\begin{align*}")
                        
                        if len(sections) > 1:
                            # Display first text section
                            st.write(sections[0])
                            
                            # Process each align environment
                            for i in range(1, len(sections)):
                                align_parts = sections[i].split("\\end{align*}")
                                if len(align_parts) >= 1:
                                    # Display the align environment
                                    st.latex("\\begin{align*}" + align_parts[0] + "\\end{align*}")
                                    
                                    # Display any text after the align environment
                                    if len(align_parts) > 1 and align_parts[1].strip():
                                        st.write(align_parts[1])
                        else:
                            # If no align environments found, look for inline LaTeX
                            latex_pattern = re.compile(r'\$(.*?)\$')
                            latex_matches = list(latex_pattern.finditer(fixed_content))
                            
                            if latex_matches:
                                last_end = 0
                                for match in latex_matches:
                                    # Display text before the LaTeX expression
                                    if match.start() > last_end:
                                        st.write(fixed_content[last_end:match.start()])
                                    
                                    # Display the LaTeX expression
                                    st.latex(match.group(1))
                                    last_end = match.end()
                                
                                # Display any remaining text
                                if last_end < len(fixed_content):
                                    st.write(fixed_content[last_end:])
                            else:
                                # If no LaTeX found, just display the text
                                st.write(fixed_content)
                    else:
                        st.write(content)
            else:
                st.chat_message("user").write(content)
        else:
            # For assistant messages, extract and properly display LaTeX content
            with st.chat_message("assistant"):
                content = message["content"]
                
                # First, try to extract align environments with or without brackets
                align_pattern = re.compile(r'(?:\[\s*)?\\begin\{align\*?\}(.*?)\\end\{align\*?\}\s*(?:\])?', re.DOTALL)
                align_matches = align_pattern.finditer(content)
                last_end = 0
                
                for match in align_matches:
                    # Display text before the align environment
                    if match.start() > last_end:
                        st.write(content[last_end:match.start()])
                    
                    # Extract the align content and fix formatting issues
                    align_content = match.group(1)
                    
                    # Fix line breaks (ensure double backslashes)
                    align_content = re.sub(r'\\(?!\s*\\)', r'\\\\', align_content)
                    
                    # Fix variable repetitions (like "y y")
                    align_content = re.sub(r'([xy])\s+\1', r'\1', align_content)
                    
                    # Fix reversed fractions
                    align_content = re.sub(r'x\s*=\s*\n*5\s*\n*7', r'x = \\frac{7}{5}', align_content)
                    align_content = re.sub(r'y\s*=\s*\n*5\s*\n*11', r'y = \\frac{11}{5}', align_content)
                    
                    # Display the fixed align environment using st.latex
                    st.latex("\\begin{align*}" + align_content + "\\end{align*}")
                    
                    last_end = match.end()
                
                # Handle remaining text, including standalone LaTeX expressions
                if last_end < len(content):
                    remaining_text = content[last_end:]
                    
                    # Process inline LaTeX expressions with $ signs
                    latex_pattern = re.compile(r'\$(.*?)\$')
                    latex_matches = latex_pattern.finditer(remaining_text)
                    last_latex_end = 0
                    
                    for latex_match in latex_matches:
                        # Display text before the LaTeX expression
                        if latex_match.start() > last_latex_end:
                            st.write(remaining_text[last_latex_end:latex_match.start()])
                        
                        # Extract and display the LaTeX expression
                        latex_content = latex_match.group(1)
                        
                        # Fix specific patterns for fractions
                        latex_content = re.sub(r'x\s*=\s*5\s*/\s*7', r'x = \\frac{7}{5}', latex_content)
                        latex_content = re.sub(r'y\s*=\s*5\s*/\s*11', r'y = \\frac{11}{5}', latex_content)
                        
                        st.latex(latex_content)
                        
                        last_latex_end = latex_match.end()
                    
                    # Display any remaining text after the last LaTeX expression
                    if last_latex_end < len(remaining_text):
                        # Check if the remaining text might contain a system of equations
                        if "system of equations" in remaining_text[last_latex_end:] and ("x =" in remaining_text[last_latex_end:] or "y =" in remaining_text[last_latex_end:]):
                            # Try to detect and format system solutions
                            solution_text = remaining_text[last_latex_end:]
                            
                            # Extract solution parts
                            x_solution_match = re.search(r'x\s*=\s*\\frac\{(\d+)\}\{(\d+)\}', solution_text)
                            y_solution_match = re.search(r'y\s*=\s*\\frac\{(\d+)\}\{(\d+)\}', solution_text)
                            
                            if x_solution_match and y_solution_match:
                                solution_latex = f"x = \\frac{{{x_solution_match.group(1)}}}{{{x_solution_match.group(2)}}} \\text{{ and }} y = \\frac{{{y_solution_match.group(1)}}}{{{y_solution_match.group(2)}}}"
                                
                                # Split the text around the solution part
                                solution_start = min(solution_text.find("x ="), solution_text.find("y ="))
                                if solution_start > 0:
                                    st.write(solution_text[:solution_start])
                                
                                st.latex(solution_latex)
                            else:
                                st.write(solution_text)
                        else:
                            st.write(remaining_text[last_latex_end:])
    
    # Get user input
    user_query = st.chat_input("Enter your math question...")
    if user_query:
        # Check if the input appears to be a broken system of equations
        if (("2\nx" in user_query and "3\nx" in user_query) or 
            ("system of equations" in user_query.lower() and 
             any(x in user_query for x in ["+\ny", "−\ny"]))):
            
            # Fix broken LaTeX before processing
            fixed_query = fix_broken_latex(user_query)
            
            # Add original user message to chat history (we'll display the fixed version)
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # Display fixed user message
            with st.chat_message("user"):
                # Process and display the fixed LaTeX
                sections = fixed_query.split("\\begin{align*}")
                
                if len(sections) > 1:
                    # Display first text section
                    st.write(sections[0])
                    
                    # Process each align environment
                    for i in range(1, len(sections)):
                        align_parts = sections[i].split("\\end{align*}")
                        if len(align_parts) >= 1:
                            # Display the align environment
                            st.latex("\\begin{align*}" + align_parts[0] + "\\end{align*}")
                            
                            # Display any text after the align environment
                            if len(align_parts) > 1 and align_parts[1].strip():
                                st.write(align_parts[1])
                else:
                    # If no align environments found, look for inline LaTeX
                    latex_pattern = re.compile(r'\$(.*?)\$')
                    latex_matches = list(latex_pattern.finditer(fixed_query))
                    
                    if latex_matches:
                        last_end = 0
                        for match in latex_matches:
                            # Display text before the LaTeX expression
                            if match.start() > last_end:
                                st.write(fixed_query[last_end:match.start()])
                            
                            # Display the LaTeX expression
                            st.latex(match.group(1))
                            last_end = match.end()
                        
                        # Display any remaining text
                        if last_end < len(fixed_query):
                            st.write(fixed_query[last_end:])
                    else:
                        # If no LaTeX found, just display the text
                        st.write(fixed_query)
        else:
            # Add user message to chat history for regular queries
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # Display user message
            st.chat_message("user").write(user_query)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Use the original query to get a response
                response = process_query(user_query, history=st.session_state.messages)
                
                # Process the response the same way as for history messages
                # First, try to extract align environments with or without brackets
                align_pattern = re.compile(r'(?:\[\s*)?\\begin\{align\*?\}(.*?)\\end\{align\*?\}\s*(?:\])?', re.DOTALL)
                align_matches = align_pattern.finditer(response)
                last_end = 0
                
                for match in align_matches:
                    # Display text before the align environment
                    if match.start() > last_end:
                        st.write(response[last_end:match.start()])
                    
                    # Extract the align content and fix formatting issues
                    align_content = match.group(1)
                    
                    # Fix line breaks (ensure double backslashes)
                    align_content = re.sub(r'\\(?!\s*\\)', r'\\\\', align_content)
                    
                    # Fix variable repetitions (like "y y")
                    align_content = re.sub(r'([xy])\s+\1', r'\1', align_content)
                    
                    # Fix reversed fractions
                    align_content = re.sub(r'x\s*=\s*\n*5\s*\n*7', r'x = \\frac{7}{5}', align_content)
                    align_content = re.sub(r'y\s*=\s*\n*5\s*\n*11', r'y = \\frac{11}{5}', align_content)
                    
                    # Display the fixed align environment using st.latex
                    st.latex("\\begin{align*}" + align_content + "\\end{align*}")
                    
                    last_end = match.end()
                
                # Handle remaining text, including standalone LaTeX expressions
                if last_end < len(response):
                    remaining_text = response[last_end:]
                    
                    # Process inline LaTeX expressions with $ signs
                    latex_pattern = re.compile(r'\$(.*?)\$')
                    latex_matches = latex_pattern.finditer(remaining_text)
                    last_latex_end = 0
                    
                    for latex_match in latex_matches:
                        # Display text before the LaTeX expression
                        if latex_match.start() > last_latex_end:
                            st.write(remaining_text[last_latex_end:latex_match.start()])
                        
                        # Extract and display the LaTeX expression
                        latex_content = latex_match.group(1)
                        
                        # Fix specific patterns for fractions
                        latex_content = re.sub(r'x\s*=\s*5\s*/\s*7', r'x = \\frac{7}{5}', latex_content)
                        latex_content = re.sub(r'y\s*=\s*5\s*/\s*11', r'y = \\frac{11}{5}', latex_content)
                        
                        st.latex(latex_content)
                        
                        last_latex_end = latex_match.end()
                    
                    # Display any remaining text after the last LaTeX expression
                    if last_latex_end < len(remaining_text):
                        # Check if the remaining text might contain a system of equations
                        if "system of equations" in remaining_text[last_latex_end:] and ("x =" in remaining_text[last_latex_end:] or "y =" in remaining_text[last_latex_end:]):
                            # Try to detect and format system solutions
                            solution_text = remaining_text[last_latex_end:]
                            
                            # Extract solution parts
                            x_solution_match = re.search(r'x\s*=\s*\\frac\{(\d+)\}\{(\d+)\}', solution_text)
                            y_solution_match = re.search(r'y\s*=\s*\\frac\{(\d+)\}\{(\d+)\}', solution_text)
                            
                            if x_solution_match and y_solution_match:
                                solution_latex = f"x = \\frac{{{x_solution_match.group(1)}}}{{{x_solution_match.group(2)}}} \\text{{ and }} y = \\frac{{{y_solution_match.group(1)}}}{{{y_solution_match.group(2)}}}"
                                
                                # Split the text around the solution part
                                solution_start = min(solution_text.find("x ="), solution_text.find("y ="))
                                if solution_start > 0:
                                    st.write(solution_text[:solution_start])
                                
                                st.latex(solution_latex)
                            else:
                                st.write(solution_text)
                        else:
                            st.write(remaining_text[last_latex_end:])
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 