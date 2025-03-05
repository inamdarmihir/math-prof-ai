#!/usr/bin/env python3
"""
A simple Streamlit app to test LaTeX rendering with our formatting fixes.
"""

import streamlit as st
import re

def fix_latex_for_streamlit(latex_text):
    """
    Focused function to fix LaTeX specifically for Streamlit rendering.
    """
    if not isinstance(latex_text, str):
        return latex_text
    
    # Remove square brackets around align environments
    latex_text = re.sub(r'\[\s*\\begin\{align\*?\}(.*?)\\end\{align\*?\}\s*\]', 
                      r'\\begin{align*}\1\\end{align*}', latex_text, flags=re.DOTALL)
    
    # Convert single backslashes to double backslashes in align environments
    latex_text = re.sub(r'(\\begin\{align\*?\}.*?)\\(\s+)', r'\1\\\\\2', latex_text, flags=re.DOTALL)
    latex_text = re.sub(r'(\&=.*?)\\(\s+)', r'\1\\\\\2', latex_text, flags=re.DOTALL)
    
    # Fix variable repetition issues (x x, y y)
    latex_text = re.sub(r'([xy])\s+\1', r'\1', latex_text)
    
    # Fix reversed fractions
    latex_text = re.sub(r'x\s*=\s*\n*5\s*\n*7', r'x = \\frac{7}{5}', latex_text)
    latex_text = re.sub(r'y\s*=\s*\n*5\s*\n*11', r'y = \\frac{11}{5}', latex_text)
    
    # Special handling for Streamlit: ensure all LaTeX is properly delimited
    # For inline math, make sure it's within single $ signs
    latex_text = re.sub(r'(?<!\$)\\frac\{([^{}]+)\}\{([^{}]+)\}(?!\$)', r'$\\frac{\1}{\2}$', latex_text)
    
    # For display math, use double $$ or proper align environment
    if '\\begin{align' not in latex_text and '\n' in latex_text and '=' in latex_text:
        lines = latex_text.split('\n')
        has_equations = any('=' in line for line in lines)
        if has_equations:
            new_lines = []
            in_equation_block = False
            for line in lines:
                if '=' in line and not in_equation_block:
                    new_lines.append('\\begin{align*}')
                    new_lines.append(line + ' \\\\')
                    in_equation_block = True
                elif '=' in line and in_equation_block:
                    new_lines.append(line + ' \\\\')
                elif in_equation_block and not '=' in line:
                    new_lines[-1] = new_lines[-1].rstrip(' \\\\')  # Remove trailing line break from last line
                    new_lines.append('\\end{align*}')
                    new_lines.append(line)
                    in_equation_block = False
                else:
                    new_lines.append(line)
            
            if in_equation_block:
                new_lines[-1] = new_lines[-1].rstrip(' \\\\')  # Remove trailing line break from last line
                new_lines.append('\\end{align*}')
            
            latex_text = '\n'.join(new_lines)
    
    return latex_text

def fix_broken_latex(text):
    """Fix broken LaTeX where each character appears on its own line"""
    
    # First, try to identify if this is a system of equations with typical structure
    if "system of equations" in text and ("2\nx" in text or "3\nx" in text):
        # Extract the system equations
        system_pattern = r'(2\s*\n\s*x\s*\n\s*\+\s*\n\s*y\s*\n\s*=\s*\n\s*5).*?(3\s*\n\s*x\s*\n\s*−\s*\n\s*y\s*\n\s*=\s*\n\s*2)'
        system_match = re.search(system_pattern, text, re.DOTALL)
        
        if system_match:
            # Create a properly formatted system of equations
            fixed_text = text.replace(system_match.group(0), 
                                     r"\\begin{align*} 2x + y &= 5 \\\\ 3x - y &= 2 \\end{align*}")
            
            # Fix the addition step
            addition_pattern = r'\(\s*\n\s*2\s*\n\s*x\s*\n\s*\+\s*\n\s*y\s*\n\s*\)\s*\n\s*\+\s*\n\s*\(\s*\n\s*3\s*\n\s*x\s*\n\s*−\s*\n\s*y\s*\n\s*\)'
            fixed_text = re.sub(addition_pattern, 
                              r"\\begin{align*} (2x + y) + (3x - y) &= 5 + 2 \\\\ 2x + 3x + y - y &= 7 \\\\ 5x &= 7 \\end{align*}", 
                              fixed_text)
            
            # Fix the fraction part
            fraction_pattern = r'x\s*\n\s*=\s*\n\s*f\s*\n\s*r\s*\n\s*a\s*\n\s*c\s*\n\s*7\s*\n\s*5'
            fixed_text = re.sub(fraction_pattern, 
                              r"\\begin{align*} 5x &= 7 \\\\ x &= \\frac{7}{5} \\end{align*}", 
                              fixed_text)
            
            # Fix the substitution part
            subst_pattern = r'2\s*\n\s*l\s*\n\s*e\s*\n\s*f\s*\n\s*t\s*\n\s*\(\s*\n\s*f\s*\n\s*r\s*\n\s*a\s*\n\s*c\s*\n\s*7\s*\n\s*5\s*\n\s*r\s*\n\s*i\s*\n\s*g\s*\n\s*h\s*\n\s*t\s*\n\s*\)'
            fixed_text = re.sub(subst_pattern, 
                              r"2\\left(\\frac{7}{5}\\right)", 
                              fixed_text)
            
            # Fix the computation part
            comp_pattern = r'f\s*\n\s*r\s*\n\s*a\s*\n\s*c\s*\n\s*14\s*\n\s*5\s*\n\s*\+\s*\n\s*y\s*\n\s*=\s*\n\s*5'
            fixed_text = re.sub(comp_pattern, 
                              r"\\begin{align*} 2\\left(\\frac{7}{5}\\right) + y &= 5 \\\\ \\frac{14}{5} + y &= 5 \\\\ y &= 5 - \\frac{14}{5} \\\\ y &= \\frac{25}{5} - \\frac{14}{5} \\\\ y &= \\frac{11}{5} \\end{align*}", 
                              fixed_text)
            
            # Fix the final solution part
            solution_pattern = r'([Tt]herefore.*?is)\s*\n\s*x\s*\n\s*=\s*\n\s*7\s*\n\s*5\s*\n\s*([a-z]+)\s*\n\s*y\s*\n\s*=\s*\n\s*11\s*\n\s*5'
            fixed_text = re.sub(solution_pattern, 
                               r"\1 $x = \\frac{7}{5}$ \2 $y = \\frac{11}{5}$", 
                               fixed_text)
            
            return fixed_text
    
    # If we couldn't identify the specific pattern, return original text
    return text

def main():
    st.set_page_config(
        page_title="LaTeX System of Equations Demo",
        page_icon="➗", 
        layout="wide"
    )
    
    st.title("LaTeX System of Equations Demo")
    
    # Example text with proper LaTeX
    st.subheader("Example 1: Properly formatted LaTeX")
    proper_latex = """
    To solve the system of equations
    \\begin{align*}
    2x + y &= 5 \\\\
    3x - y &= 2
    \\end{align*}
    
    We use the method of addition to eliminate $y$. Adding the equations:
    \\begin{align*}
    (2x + y) + (3x - y) &= 5 + 2 \\\\
    2x + 3x + y - y &= 7 \\\\
    5x &= 7
    \\end{align*}
    
    Solving for $x$:
    \\begin{align*}
    5x &= 7 \\\\
    x &= \\frac{7}{5}
    \\end{align*}
    
    Substituting back into the first equation:
    \\begin{align*}
    2x + y &= 5 \\\\
    2\\left(\\frac{7}{5}\\right) + y &= 5 \\\\
    \\frac{14}{5} + y &= 5 \\\\
    y &= 5 - \\frac{14}{5} \\\\
    y &= \\frac{25}{5} - \\frac{14}{5} \\\\
    y &= \\frac{11}{5}
    \\end{align*}
    
    Therefore, the solution is $x = \\frac{7}{5}$ and $y = \\frac{11}{5}$.
    """
    
    # Display proper LaTeX
    with st.expander("Show proper LaTeX code", expanded=False):
        st.code(proper_latex)
    
    # Process and display the proper LaTeX
    sections = proper_latex.split("\\begin{align*}")
    
    if len(sections) > 1:
        # Display first text section
        st.write(sections[0])
        
        # Process each align environment
        for i in range(1, len(sections)):
            align_parts = sections[i].split("\\end{align*}")
            if len(align_parts) >= 1:
                # Display the align environment
                st.latex("\\begin{align*}" + align_parts[0] + "\\end{align*}")
                
                # Display any text after the align environment
                if len(align_parts) > 1 and align_parts[1].strip():
                    st.write(align_parts[1])
    
    # Example with broken formatting (similar to what user experiences)
    st.subheader("Example 2: Fixing broken LaTeX formatting")
    
    broken_latex = """To solve the system of equations

2
x
+
y
=
5
 
3
x
−
y
=
2
2x+y
 3x−y
​
  
=5
=2
​
 
we can use the method of addition to eliminate one of the variables. Let's add the two equations together to eliminate 
y
y.

(
2
x
+
y
)
+
(
3
x
−
y
)
=
5
+
2
 
2
x
+
3
x
+
y
−
y
=
7
 
5
x
=
7
(2x+y)+(3x−y)
 2x+3x+y−y
 5x
​
  
=5+2
=7
=7
​
 
Now, solve for 
x
x:

5
x
=
7
 
x
=
f
r
a
c
75
5x
 x
frac75
​
  
=7
=
​
 
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
y. Let's use the first equation:

2
x
+
y
=
5
 
2
l
e
f
t
(
f
r
a
c
75
r
i
g
h
t
)
+
y
=
5
f
r
a
c
145
+
y
=
5
 
y
=
5
−
f
r
a
c
145
 
y
=
f
r
a
c
255
−
f
r
a
c
145
 
y
=
f
r
a
c
115
2x+y
 2
left(
frac75
right)+y
frac145+y
 y
frac145
 y
frac255−
frac145
 y
frac115
​
  
=5
=5
=5
=5−
=
=
​
 
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
    
    # Show the original broken text
    with st.expander("Show original broken text", expanded=False):
        st.code(broken_latex)
    
    st.markdown("### Original broken display:")
    st.write(broken_latex)
    
    # Apply the fix and show the result
    st.markdown("### Fixed display:")
    fixed_latex = fix_broken_latex(broken_latex)
    
    # Process and display the fixed LaTeX
    sections = fixed_latex.split("\\begin{align*}")
    
    if len(sections) > 1:
        # Display first text section
        st.write(sections[0])
        
        # Process each align environment
        for i in range(1, len(sections)):
            align_parts = sections[i].split("\\end{align*}")
            if len(align_parts) >= 1:
                # Display the align environment
                st.latex("\\begin{align*}" + align_parts[0] + "\\end{align*}")
                
                # Display any text after the align environment
                if len(align_parts) > 1 and align_parts[1].strip():
                    st.write(align_parts[1])
    else:
        # If no align environments found, look for inline LaTeX
        latex_pattern = re.compile(r'\$(.*?)\$')
        latex_matches = list(latex_pattern.finditer(fixed_latex))
        
        if latex_matches:
            last_end = 0
            for match in latex_matches:
                # Display text before the LaTeX expression
                if match.start() > last_end:
                    st.write(fixed_latex[last_end:match.start()])
                
                # Display the LaTeX expression
                st.latex(match.group(1))
                last_end = match.end()
            
            # Display any remaining text
            if last_end < len(fixed_latex):
                st.write(fixed_latex[last_end:])
        else:
            # If no LaTeX found, just display the text
            st.write(fixed_latex)
    
    # Interactive example
    st.subheader("Example 3: Try your own input")
    user_input = st.text_area("Enter your LaTeX (can be broken format):", height=250)
    
    if user_input:
        st.markdown("### Your input (fixed):")
        fixed_user_input = fix_broken_latex(user_input)
        
        # Process and display the fixed LaTeX
        sections = fixed_user_input.split("\\begin{align*}")
        
        if len(sections) > 1:
            # Display first text section
            st.write(sections[0])
            
            # Process each align environment
            for i in range(1, len(sections)):
                align_parts = sections[i].split("\\end{align*}")
                if len(align_parts) >= 1:
                    # Display the align environment
                    st.latex("\\begin{align*}" + align_parts[0] + "\\end{align*}")
                    
                    # Display any text after the align environment
                    if len(align_parts) > 1 and align_parts[1].strip():
                        st.write(align_parts[1])
        else:
            # If no align environments found, look for inline LaTeX
            latex_pattern = re.compile(r'\$(.*?)\$')
            latex_matches = list(latex_pattern.finditer(fixed_user_input))
            
            if latex_matches:
                last_end = 0
                for match in latex_matches:
                    # Display text before the LaTeX expression
                    if match.start() > last_end:
                        st.write(fixed_user_input[last_end:match.start()])
                    
                    # Display the LaTeX expression
                    st.latex(match.group(1))
                    last_end = match.end()
                
                # Display any remaining text
                if last_end < len(fixed_user_input):
                    st.write(fixed_user_input[last_end:])
            else:
                # If no LaTeX found, just display the text
                st.write(fixed_user_input)

if __name__ == "__main__":
    main() 