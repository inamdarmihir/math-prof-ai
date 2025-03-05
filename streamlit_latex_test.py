import streamlit as st
import math_agent

st.title("LaTeX Formatting Test")
st.write("This app tests the improved LaTeX formatting function.")

# Test examples
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
    st.subheader(f"Example {i+1}")
    
    # Input section
    st.write("**Input:**")
    st.code(example, language="latex")
    
    # Process the example
    result = math_agent.fix_latex_formatting(example)
    
    # Output section
    st.write("**Output (code):**")
    st.code(result, language="latex")
    
    # Rendered output
    st.write("**Rendered Output:**")
    st.latex(result)
    
    st.markdown("---")

# Add a custom input section
st.subheader("Try your own LaTeX")
user_input = st.text_area("Enter LaTeX code:", value="\\begin{align} E &= mc^2 \\\\ F &= ma \\end{align}")

if user_input:
    # Process the user input
    result = math_agent.fix_latex_formatting(user_input)
    
    # Show the processed code
    st.write("**Processed Output (code):**")
    st.code(result, language="latex")
    
    # Show the rendered result
    st.write("**Rendered Output:**")
    st.latex(result) 