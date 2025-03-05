import math_agent

# Test the square brackets around align environment example
example = "[\\begin{align} \\vec{F} &= m\\vec{a} \\\\ \\vec{p} &= m\\vec{v} \\end{align}]"

print("Input:")
print(example)
print("\nOutput:")
result = math_agent.fix_latex_formatting(example)
print(result) 