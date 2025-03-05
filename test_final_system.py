#!/usr/bin/env python3
"""
Final test script for the exact system of equations example
"""
import math_agent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The exact LaTeX code provided by the user with all the issues
test_case = r"""To solve the system of equations \begin{align*} 2x + y &= 5 \
3x - 
y
y &= 2 \end{align*} we can use the method of addition to eliminate one of the variables. Let's add the two equations together to eliminate 
y
y.

\begin{align*} (2x + y) + (3x - 
y
y) &= 5 + 2 \
2x + 3x + 
y
y - 
y
y &= 7 \ 5x &= 7 \end{align*}

Now, solve for 
x
x: \begin{align*} 5x &= 7 \
x &= \frac{7}{5} \end{align*}

With 
x
=
7
5
x= 
5
7
​
 , we can substitute this value back into one of the original equations to solve for 
y
y. Let's use the first equation: \begin{align*} 2x + y &= 5 \
2\left(\frac{7}{5}\right) + 
y
y &= 5 \ \frac{14}{5} + 
y
y &= 5 \ 
y
y &= 5 - \frac{14}{5} \ 
y
y &= \frac{25}{5} - \frac{14}{5} \ 
y
y &= \frac{11}{5} \end{align*}

Therefore, the solution to the system of equations is 
x
=
7
5
x= 
5
7
​
  and 
y
=
11
5
y= 
5
11
​
 ."""

def test_final_fix():
    """Test the final LaTeX formatting fixes on the exact user example."""
    logger.info("Testing final fixes on system of equations example...")
    
    # Apply fix_latex_formatting
    fixed = math_agent.fix_latex_formatting(test_case)
    
    # Print the result
    logger.info("\nFixed output:")
    print(fixed)
    
    # Verify key improvements
    
    # 1. Check for doubled "y y" patterns - should be fixed
    assert "y y" not in fixed, "Duplicate 'y y' pattern still exists"
    
    # 2. Check that single backslashes are converted to double backslashes
    assert '\\\\' in fixed, "Double backslashes not found in output"
    
    # 3. Check that fractions are correctly formatted (not reversed)
    assert '\\frac{7}{5}' in fixed, "Fraction 7/5 not found or incorrect"
    assert '\\frac{11}{5}' in fixed, "Fraction 11/5 not found or incorrect"
    assert '\\frac{5}{7}' not in fixed, "Reversed fraction 5/7 found - should be 7/5"
    assert '\\frac{5}{11}' not in fixed, "Reversed fraction 5/11 found - should be 11/5"
    
    # 4. Check for correct math mode formatting
    assert "$x$" in fixed or "\\begin{align" in fixed, "Variable x not properly formatted"
    assert "$y$" in fixed or "\\begin{align" in fixed, "Variable y not properly formatted"
    
    logger.info("All tests passed! The LaTeX has been properly fixed.")

if __name__ == "__main__":
    test_final_fix() 