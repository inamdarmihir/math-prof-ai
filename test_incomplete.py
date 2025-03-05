import math_agent

# Test the incomplete align environment example
example = "\\begin{align} f(x) &= x^2 - 4x + 4 \\\\ f'(x) &= 2x - 4"

print("Input:")
print(example)
print("\nOutput:")
result = math_agent.fix_latex_formatting(example)
print(result) 