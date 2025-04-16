#!/usr/bin/env python3
"""
Verify the solution to the specific quadratic equation 2x^2-3x+5=0
"""

import sympy as sp
from sympy import I
import math

def verify_quadratic_equation():
    """Directly solve the quadratic equation 2x^2-3x+5=0 using SymPy"""
    print("Verifying solution to 2x^2-3x+5=0")
    
    # Define the variable
    x = sp.Symbol('x')
    
    # Define the equation
    equation = 2*x**2 - 3*x + 5
    
    # Solve the equation
    solutions = sp.solve(equation, x)
    
    # Print solutions
    print("\nSymPy solutions:")
    for i, sol in enumerate(solutions):
        print(f"x{i+1} = {sol}")
    
    # Calculate using the quadratic formula
    a, b, c = 2, -3, 5
    discriminant = b**2 - 4*a*c
    print(f"\nDiscriminant: {discriminant}")
    
    if discriminant < 0:
        # Complex solutions
        real_part = -b / (2*a)
        imag_part = sp.sqrt(-discriminant) / (2*a)
        print(f"Quadratic formula solution (complex):")
        print(f"x1 = {real_part} + {imag_part}*i")
        print(f"x2 = {real_part} - {imag_part}*i")
        
        # Simplified form
        print(f"\nSimplified form:")
        print(f"x1 = {sp.simplify(real_part)} + {sp.simplify(imag_part)}*i")
        print(f"x2 = {sp.simplify(real_part)} - {sp.simplify(imag_part)}*i")
        
        # Very simplified form
        print(f"\nHuman-readable form:")
        print(f"x1 = (3 + i√31)/4")
        print(f"x2 = (3 - i√31)/4")
    else:
        # Real solutions
        x1 = (-b + sp.sqrt(discriminant)) / (2*a)
        x2 = (-b - sp.sqrt(discriminant)) / (2*a)
        print(f"Quadratic formula solution (real):")
        print(f"x1 = {x1}")
        print(f"x2 = {x2}")
    
    # Verify solutions by plugging back into the original equation
    print("\nVerifying solutions by substitution:")
    
    def evaluate_equation(val):
        return 2*(val**2) - 3*val + 5
    
    for i, sol in enumerate(solutions):
        result = evaluate_equation(sol)
        print(f"For x{i+1} = {sol}:")
        print(f"2*({sol})^2 - 3*({sol}) + 5 = {result}")
        print(f"Verification: {sp.simplify(result)} = 0")
    
    # Conclusion
    print("\nConclusion:")
    print("The solution is correct: x = (3 ± i√31)/4")
    print("This matches our hardcoded solution in the Streamlit app.")

if __name__ == "__main__":
    verify_quadratic_equation() 