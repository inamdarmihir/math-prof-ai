import math_agent

# The problematic incomplete align environment example
example = "\\begin{align} f(x) &= x^2 - 4x + 4 \\\\ f'(x) &= 2x - 4"

print("Input:")
print(repr(example))
print("\nRaw Input:")
for idx, char in enumerate(example):
    print(f"{idx}: '{char}' (ord: {ord(char)})")

# Process using the fix_latex_formatting function
result = math_agent.fix_latex_formatting(example)

print("\nOutput:")
print(repr(result))
print("\nRaw Output:")
for idx, char in enumerate(result):
    print(f"{idx}: '{char}' (ord: {ord(char)})") 