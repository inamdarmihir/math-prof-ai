#!/usr/bin/env python3
"""
Test script for system of equations LaTeX formatting fix
"""
import math_agent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The problematic LaTeX code with single backslash line breaks
test_case = r"""To solve the system of equations \begin{align*} 2x + y &= 5 \ 3x - y &= 2 \end{align*} we can use the method of addition to eliminate one of the variables. Let's add the two equations together to eliminate 
y
y.

\begin{align*} (2x + y) + (3x - y) &= 5 + 2 \ 2x + 3x + y - y &= 7 \ 5x &= 7 \end{align*}

Now, solve for 
x
x: \begin{align*} 5x &= 7 \ x &= \frac{7}{5} \end{align*}

With 
x
=
7
5
x= 
5
7
​
 , we can substitute this value into one of the original equations to solve for 
y
y. Let's use the first equation: \begin{align*} 2x + y &= 5 \ 2\left(\frac{7}{5}\right) + y &= 5 \ \frac{14}{5} + y &= 5 \ y &= 5 - \frac{14}{5} \ y &= \frac{25}{5} - \frac{14}{5} \ y &= \frac{11}{5} \end{align*}

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

def test_system_of_equations():
    """Test the LaTeX formatting for system of equations."""
    logger.info("Testing system of equations formatting...")
    
    # Apply fix_latex_formatting
    fixed = math_agent.fix_latex_formatting(test_case)
    
    # Print the result
    logger.info("\nFixed output:")
    print(fixed)
    
    # Verify key improvements
    
    # 1. Check that single backslashes are converted to double backslashes in align environments
    assert '\\\\' in fixed, "Double backslashes not found in output"
    
    # 2. Check that fractions are correctly formatted (not reversed)
    assert '\\frac{7}{5}' in fixed, "Fraction 7/5 not found"
    assert '\\frac{11}{5}' in fixed, "Fraction 11/5 not found"
    
    # 3. Verify single variables are properly wrapped in math delimiters
    assert '$x$' in fixed or '$x ' in fixed, "Variable x not properly wrapped"
    assert '$y$' in fixed or '$y ' in fixed, "Variable y not properly wrapped"
    
    logger.info("Test passed! The LaTeX formatting has been fixed.")

if __name__ == "__main__":
    test_system_of_equations() 