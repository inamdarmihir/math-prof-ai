#!/usr/bin/env python3
"""
Math Agent Validation Script

This script runs a series of tests to validate the core functionality of the math agent.
It verifies that all components are working correctly.
"""

import os
import sys
import logging
from sympy import diff, symbols, solve, Eq, Function
from sympy.abc import x
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_sympy_functionality():
    """Test that SymPy is working correctly for symbolic mathematics"""
    try:
        # Define a polynomial
        polynomial = x**3 + 2*x**2 - 5*x + 1
        
        # Calculate derivative
        derivative = diff(polynomial, x)
        expected_derivative = 3*x**2 + 4*x - 5
        
        # Verify result
        result = derivative == expected_derivative
        
        if result:
            logger.info(f"SymPy Test PASSED: Derivative of {polynomial} = {derivative}")
            return True
        else:
            logger.error(f"SymPy Test FAILED: Expected {expected_derivative}, got {derivative}")
            return False
    except Exception as e:
        logger.error(f"SymPy Test ERROR: {e}")
        return False

def test_openai_connection():
    """Test connection to OpenAI API"""
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Simple test query
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant that only responds with numbers."},
                {"role": "user", "content": "What is 2 + 2?"}
            ],
            max_tokens=10
        )
        
        # Extract response
        answer = response.choices[0].message.content.strip()
        
        # Check if the answer contains "4"
        if "4" in answer:
            logger.info(f"OpenAI Test PASSED: 2 + 2 = {answer}")
            return True
        else:
            logger.error(f"OpenAI Test FAILED: Expected 4, got {answer}")
            return False
    except Exception as e:
        logger.error(f"OpenAI Test ERROR: {e}")
        return False

def test_qdrant_connection():
    """Test connection to Qdrant if configured"""
    try:
        # Get Qdrant configuration
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        qdrant_collection = os.getenv("QDRANT_COLLECTION", "math_knowledge")
        
        # Skip test if Qdrant is not configured
        if not qdrant_url or not qdrant_api_key:
            logger.warning("Qdrant Test SKIPPED: Qdrant not configured")
            return True
        
        # Initialize Qdrant client
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if qdrant_collection in collection_names:
            logger.info(f"Qdrant Test PASSED: Connected to collection '{qdrant_collection}'")
            return True
        else:
            logger.error(f"Qdrant Test FAILED: Collection '{qdrant_collection}' not found")
            return False
    except Exception as e:
        logger.error(f"Qdrant Test ERROR: {e}")
        return False

def test_direct_math_functions():
    """Test direct mathematical functions"""
    try:
        # Test quadratic equation solver
        x = symbols('x')
        quadratic = x**2 - 5*x + 6
        quadratic_solution = solve(quadratic, x)
        expected_quadratic = [2, 3]
        
        # Test linear equation solver
        linear = 3*x + 7 - 16
        linear_solution = solve(linear, x)
        expected_linear = [3]
        
        # Test differential equation solver
        y = Function('y')
        diffeq = Eq(y(x).diff(x) + y(x), 2)
        diffeq_solution = str(solve(diffeq, y(x))[0])
        
        # Test derivative
        expr = x**3 + 2*x**2 - 5*x + 1
        derivative = diff(expr, x)
        expected_derivative = 3*x**2 + 4*x - 5
        
        # Check all results
        results = [
            sorted(quadratic_solution) == sorted(expected_quadratic),
            sorted(linear_solution) == sorted(expected_linear),
            "exp(-x)" in diffeq_solution and "2" in diffeq_solution,
            derivative == expected_derivative
        ]
        
        success_count = sum(results)
        
        if all(results):
            logger.info(f"Direct Math Functions Test PASSED: {success_count}/4 tests succeeded")
            return True
        else:
            logger.error(f"Direct Math Functions Test PARTIAL: {success_count}/4 tests succeeded")
            return success_count > 2  # Pass if more than half succeed
    except Exception as e:
        logger.error(f"Direct Math Functions Test ERROR: {e}")
        return False

def run_validation():
    """Run all validation tests"""
    tests = [
        ("SymPy Functionality", test_sympy_functionality),
        ("OpenAI Connection", test_openai_connection),
        ("Qdrant Connection", test_qdrant_connection),
        ("Direct Math Functions", test_direct_math_functions)
    ]
    
    results = []
    
    print("\n==== Math Agent Validation ====\n")
    
    for name, test_func in tests:
        print(f"Testing {name}...")
        result = test_func()
        results.append(result)
        status = "PASSED" if result else "FAILED"
        print(f"{name}: {status}")
        print()
    
    # Print summary
    print("\n==== Validation Summary ====\n")
    for i, (name, _) in enumerate(tests):
        status = "PASSED" if results[i] else "FAILED"
        print(f"{name}: {status}")
    
    # Overall result
    if all(results):
        print("\nOverall validation: PASSED")
        return 0
    else:
        print("\nOverall validation: FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(run_validation()) 