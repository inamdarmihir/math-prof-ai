#!/usr/bin/env python3
"""
Test script for LaTeX formatting fixes
"""
import math_agent
import logging

# Configure logging for this test script
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test cases
test_cases = [
    # Properly balanced align environment
    "\\begin{align} x^2 + 2x &= 5 \\\\ x &= -1 ± \\sqrt{6} \\end{align}",
    
    # Align environment with dollar signs
    "$\\begin{align} y = mx + b \\\\ m = \\frac{y_2 - y_1}{x_2 - x_1} \\end{align}$",
    
    # Incomplete align environment
    "\\begin{align} f(x) &= x^2 - 4x + 4 \\\\ f'(x) &= 2x - 4",
    
    # Align environment with bad alignment markers
    "\\begin{align} x^2 = 4 \\\\ y = 3x + 2 \\end{align}",
    
    # Fractions with incorrect spacing
    "\\frac{ 2x^2 + 3x }{ 4x - 1 }",
    
    # Nested environments
    "\\begin{align} \\int_{0}^{1} x^2 dx &= \\left[ \\frac{x^3}{3} \\right]_{0}^{1} \\\\ &= \\frac{1}{3} \\end{align}",
    
    # Dollar sign balancing issue
    "The value of x is $\\sqrt{2}$",
    
    # Square brackets around math
    "[x^2 + y^2 = r^2]"
]

def test_fix_latex_formatting():
    """Test the LaTeX formatting improvements"""
    logger.info("Testing LaTeX formatting fixes...")
    
    for i, test_case in enumerate(test_cases):
        logger.info(f"\nTest Case {i+1}:")
        logger.info(f"INPUT: {repr(test_case)}")
        
        # Apply fix_latex_formatting
        fixed = math_agent.fix_latex_formatting(test_case)
        logger.info(f"OUTPUT: {repr(fixed)}")
        
        # Verify balanced environments
        if "\\begin{align" in fixed:
            assert "\\end{align" in fixed, "Align environment not balanced"
        
        # Verify dollar sign balance
        dollar_count = fixed.count('$')
        assert dollar_count % 2 == 0, f"Dollar sign count ({dollar_count}) is not even"
        
        logger.info(f"Test {i+1} passed successfully")
    
    logger.info("\nAll tests passed!")

if __name__ == "__main__":
    test_fix_latex_formatting() 