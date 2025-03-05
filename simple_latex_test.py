import math_agent

# Test different LaTeX examples
examples = [
    # Simple align environment
    "\\begin{align} x^2 + 2x &= 5 \\\\ x &= -1 ± \\sqrt{6} \\end{align}",
    
    # Align environment with dollar signs
    "$\\begin{align} y = mx + b \\\\ m = \\frac{y_2 - y_1}{x_2 - x_1} \\end{align}$",
    
    # Complex example with fractions and integrals
    "\\begin{align} \\frac{dx}{dt} &= \\frac{3x^2 + 2x}{4t - 1} \\\\ \\int \\frac{1}{x} dx &= \\ln|x| + C \\end{align}",
    
    # Incomplete align environment
    "\\begin{align} f(x) &= x^2 - 4x + 4 \\\\ f'(x) &= 2x - 4",
    
    # Square brackets around align
    "[\\begin{align} \\vec{F} &= m\\vec{a} \\\\ \\vec{p} &= m\\vec{v} \\end{align}]"
]

# Test each example
for i, example in enumerate(examples):
    print(f"\n--- Example {i+1} ---")
    print("\nInput:")
    print(example)
    print("\nOutput:")
    result = math_agent.fix_latex_formatting(example)
    print(result)
    print("-" * 50) 