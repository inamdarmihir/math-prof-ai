#!/usr/bin/env python3
"""
Comprehensive test for LaTeX formatting across different types of math problems.
This test ensures that our fixes work for various math notations, not just the system of equations example.
"""

import logging
import re
import sys
import math_agent
from math_agent import fix_latex_formatting

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_all_math_problems():
    """Test LaTeX formatting on various types of math problems."""
    logger.info("Testing LaTeX formatting on various types of math problems...")
    
    # Test cases for various math problems
    test_cases = [
        # Basic equations
        {
            "name": "Simple equation",
            "input": "To solve the equation $2x + 3 = 7$, subtract 3 from both sides: $2x = 4$. Then divide by 2: $x = 2$.",
            "check": lambda fixed: "$2x + 3 = 7$" in fixed and "$x = 2$" in fixed
        },
        
        # Quadratic equations
        {
            "name": "Quadratic equation",
            "input": "Solve $x^2 - 5x + 6 = 0$ using the quadratic formula: $x = \\frac{5 \\pm \\sqrt{5^2 - 4 \\cdot 1 \\cdot 6}}{2 \\cdot 1}$ = $\\frac{5 \\pm \\sqrt{25 - 24}}{2}$ = $\\frac{5 \\pm \\sqrt{1}}{2}$ = $\\frac{5 \\pm 1}{2}$. So $x = 3$ or $x = 2$.",
            "check": lambda fixed: "\\frac{5 \\pm \\sqrt{25 - 24}}{2}" in fixed and ("$x = 3$" in fixed or "$x = 2$" in fixed)
        },
        
        # Derivatives with formatting issues
        {
            "name": "Derivative",
            "input": "Calculate the derivative of $f(x) = x^2 + 3x + 2$.\n$f'(x) = 2x + 3$",
            "check": lambda fixed: "$f(x) = x^2 + 3x + 2$" in fixed and "$f'(x) = 2x + 3$" in fixed
        },
        
        # Integrals with spacing issues
        {
            "name": "Integral",
            "input": "Evaluate the integral $\\int x^2 dx = \\frac{x^3}{3} + C$",
            "check": lambda fixed: "\\int x^2 dx" in fixed and "\\frac{x^3}{3} + C" in fixed
        },
        
        # System of equations (different format than our previous example)
        {
            "name": "Alternative system of equations",
            "input": """Consider the system of equations:
            \\begin{align}
            3x + 2y &= 12\\
            2x - y &= 5
            \\end{align}
            
            Solving for $y$ in the second equation: $y = 2x - 5$
            
            Substituting into the first equation:
            $3x + 2(2x - 5) = 12$
            $3x + 4x - 10 = 12$
            $7x = 22$
            $x = 22/7$
            
            Back-substituting:
            $y = 2(22/7) - 5 = 44/7 - 5 = 44/7 - 35/7 = 9/7$
            
            Therefore, $x = 22/7$ and $y = 9/7$.""",
            "check": lambda fixed: "\\begin{align}" in fixed and ("$x = \\frac{22}{7}$" in fixed or "$x = 22/7$" in fixed)
        },
        
        # Calculus problems with align environments
        {
            "name": "Calculus with align",
            "input": """Find the critical points of $f(x) = x^3 - 3x^2 + 2$.
            \\begin{align*}
            f'(x) &= 3x^2 - 6x\\
            0 &= 3x^2 - 6x\\
            0 &= 3x(x - 2)
            \\end{align*}
            
            So $x = 0$ or $x = 2$ are the critical points.""",
            "check": lambda fixed: "\\begin{align*}" in fixed and "3x(x - 2)" in fixed and "\\end{align*}" in fixed
        },
        
        # Matrix operations
        {
            "name": "Matrix operations",
            "input": """Consider the matrix $A = \\begin{bmatrix} 1 & 2 \\ 3 & 4 \\end{bmatrix}$.
            The determinant is $|A| = 1 \\cdot 4 - 2 \\cdot 3 = 4 - 6 = -2$.""",
            "check": lambda fixed: "\\begin{bmatrix}" in fixed and "|A| = " in fixed and "-2" in fixed
        },
        
        # Fraction handling
        {
            "name": "Fractions",
            "input": "The result is $\\frac{x+1}{x-1}$ and when $x = 3$, we get $\\frac{3+1}{3-1} = \\frac{4}{2} = 2$.",
            "check": lambda fixed: "\\frac{x+1}{x-1}" in fixed and "\\frac{4}{2}" in fixed
        },
        
        # Line breaks and alignment in multi-line equations
        {
            "name": "Multi-line equations",
            "input": """Simplify the expression:
            \\begin{align*}
            (x+2)(x-3) &= x^2 - 3x + 2x - 6\\
            &= x^2 - x - 6
            \\end{align*}""",
            "check": lambda fixed: "\\begin{align*}" in fixed and "x^2 - x - 6" in fixed and "\\end{align*}" in fixed
        }
    ]
    
    # Test each case
    all_passed = True
    for i, test_case in enumerate(test_cases):
        logger.info(f"Testing case {i+1}: {test_case['name']}")
        try:
            fixed_output = fix_latex_formatting(test_case["input"])
            
            # Check if the output passes the specific test case checks
            if test_case["check"](fixed_output):
                logger.info(f"✓ Test case {i+1} ({test_case['name']}) passed!")
            else:
                logger.error(f"✗ Test case {i+1} ({test_case['name']}) failed checking criteria.")
                logger.error(f"Output: {fixed_output}")
                all_passed = False
                
            # Basic checks common to all test cases
            # 1. Check for balanced LaTeX environments
            for env in ["align", "align*", "equation", "bmatrix"]:
                begin_count = fixed_output.count(f"\\begin{{{env}}}")
                end_count = fixed_output.count(f"\\end{{{env}}}")
                if begin_count != end_count:
                    logger.error(f"✗ Unbalanced {env} environment: {begin_count} begins, {end_count} ends")
                    all_passed = False
            
            # 2. Check for balanced dollar signs
            dollar_count = fixed_output.count('$')
            if dollar_count % 2 != 0:
                logger.error(f"✗ Unbalanced dollar signs: {dollar_count} $ found")
                all_passed = False
                
            # 3. Check for common LaTeX errors
            if "\\\\end" in fixed_output or "\\\\begin" in fixed_output:
                logger.error("✗ Double backslash before begin/end command")
                all_passed = False
                
            # 4. Check that line breaks in align environments use double backslashes
            align_blocks = re.findall(r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}', fixed_output, re.DOTALL)
            for block in align_blocks:
                if re.search(r'[^\\]\\(?!\\\s|\s)', block):
                    logger.error(f"✗ Possible single backslash as line break in align environment: {block}")
                    all_passed = False
            
        except Exception as e:
            logger.error(f"✗ Test case {i+1} ({test_case['name']}) raised an exception: {str(e)}")
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    logger.info("Starting comprehensive LaTeX formatting tests...")
    if test_all_math_problems():
        logger.info("✓ All tests passed! LaTeX formatting works for various math problems.")
        sys.exit(0)
    else:
        logger.error("✗ Some tests failed. See above for details.")
        sys.exit(1) 