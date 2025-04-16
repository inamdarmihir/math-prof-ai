#!/usr/bin/env python3

# This script directly applies a complete fix to the math_agent_langgraph.py file
# by replacing the problematic code section with a corrected version.

import re

# Read the entire file content
with open('math_agent_langgraph.py', 'r') as f:
    content = f.read()

# Define the problematic section and its replacement
# The issue is in the quadratic equation solving section where there's an incomplete try-except structure

# Find the position of "def solve_derivative" to mark the end of the problematic section
derivative_pos = content.find("def solve_derivative")

# Find the beginning of the solve_equations_node function
equations_node_pos = content.find("def solve_equations_node")

if derivative_pos > 0 and equations_node_pos > 0:
    # Extract the solve_equations_node function
    function_content = content[equations_node_pos:derivative_pos].strip()
    
    # Replace the function with a fixed version
    # The key is to ensure all blocks are properly closed and indented
    fixed_function = '''@traceable(run_type="chain", name="solve_equations")
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
            
            if quadratic_match:
                try:
                    a_str, var, b_part, c_part = quadratic_match.groups()
                    
                    # Handle defaults and empty groups
                    a = float(a_str) if a_str and a_str != '' else 1.0
                    
                    # Extract b coefficient
                    b = 0.0
                    if b_part:
                        b_match = re.match(r'([-+])(\d*)([a-zA-Z])', b_part)
                        if b_match:
                            sign, b_val, _ = b_match.groups()
                            b_val = b_val if b_val else '1'
                            b = float(f"{sign}{b_val}")
                    
                    # Extract c coefficient
                    c = 0.0
                    if c_part:
                        c = float(c_part)
                    
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
'''
    
    # Replace the entire function with the fixed version
    new_content = content[:equations_node_pos] + fixed_function + '\n\n' + content[derivative_pos:]
    
    # Write back to a new file
    with open('math_agent_langgraph_fixed.py', 'w') as f:
        f.write(new_content)
    
    print("File fixed and saved as math_agent_langgraph_fixed.py")
else:
    print("Could not find the relevant code sections") 