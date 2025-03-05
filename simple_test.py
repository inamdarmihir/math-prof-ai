import math_agent

# Test the dollar sign around align environments case
print("Testing dollar sign around align environments")
test = "$\\begin{align} x^2 + 2x &= 5 \\\\ y &= 3x \\end{align}$"
print("Before:")
print(test)
result = math_agent.fix_latex_formatting(test)
print("\nAfter:")
print(result)
print("-" * 50)

# Test incomplete align environment
print("\nTesting incomplete align environment")
test = "\\begin{align} x^2 + 2x &= 5 \\\\ y &= 3x"
print("Before:")
print(test)
result = math_agent.fix_latex_formatting(test)
print("\nAfter:")
print(result)
print("-" * 50)

# Test output_guardrails function
print("\nTesting output_guardrails function")
test = "The solution to the equation is \\begin{align} x^2 + 2x &= 5 \\\\ x &= -1 ± \\sqrt{6} \\end{align} as we can verify."
print("Before:")
print(test)
result = math_agent.output_guardrails(test)
print("\nAfter:")
print(result)
print("-" * 50) 