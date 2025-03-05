#!/usr/bin/env python3
"""
Enhanced LaTeX formatting fix for math_agent.py
"""
import os
import re

def fix_latex_format_function(file_path):
    """
    Improves the fix_latex_formatting function to handle alignment environments better
    
    Args:
        file_path (str): Path to the math_agent.py file
    """
    # Create backup
    backup_file = f"{file_path}.latex_backup"
    if not os.path.exists(backup_file):
        with open(file_path, 'r', encoding='utf-8') as src:
            with open(backup_file, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        print(f"Created backup at {backup_file}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the fix_latex_formatting function
    latex_function_pattern = r'def fix_latex_formatting\(text\):(.*?)(?=def [a-zA-Z_]+\()'
    match = re.search(latex_function_pattern, content, re.DOTALL)
    
    if match:
        original_function = match.group(0)
        
        # Create an enhanced version of the function
        enhanced_function = '''def fix_latex_formatting(text):
    """Fix common LaTeX formatting issues in the Math Agent output.
    
    This function addresses several common formatting problems:
    1. Removes unnecessary square brackets around align environments
    2. Fixes backslash spacing issues
    3. Ensures proper delimiters for block and inline equations
    4. Corrects alignment environment formatting
    
    Args:
        text (str): The text containing LaTeX equations
        
    Returns:
        str: Text with properly formatted LaTeX
    """
    if not isinstance(text, str):
        # If text is not a string (e.g., list, None), convert it to string
        return str(text)
        
    # Fix align environments with square brackets
    text = re.sub(r'\[\s*\\begin\{align\*?\}(.*?)\\end\{align\*?\}\s*\]', 
                 r'$$\\begin{align}\1\\end{align}$$', 
                 text, flags=re.DOTALL)
    
    # Fix standalone align environments (without brackets)
    text = re.sub(r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}', 
                 r'$$\\begin{align}\1\\end{align}$$', 
                 text, flags=re.DOTALL)
    
    # Fix alignment syntax for &= in align environments
    text = re.sub(r'(\\begin\{align\}.*?)(\d+[a-zA-Z]*\s*[\+\-\*\/].*?)\s*&=\s*', 
                 r'\1\2 &= ', 
                 text, flags=re.DOTALL)
    
    # Fix incorrect line breaks in align environments
    text = re.sub(r'\\\\(\s+)', r'\\\\ \n', text)
    
    # Fix &= spacing issues
    text = re.sub(r'&=\s+\\', r'&= \\', text)
    
    # Fix extraneous closing brackets after align environments
    text = re.sub(r'\\end\{align\}\$\$\s*\\]', r'\\end{align}$$', text)
    
    # Fix extra spaces around operators
    text = re.sub(r'(\d+)\s*\+\s*(\d+)', r'\1 + \2', text)
    text = re.sub(r'(\d+)\s*\-\s*(\d+)', r'\1 - \2', text)
    text = re.sub(r'(\d+)\s*\*\s*(\d+)', r'\1 * \2', text)
    text = re.sub(r'(\d+)\s*\/\s*(\d+)', r'\1 / \2', text)
    
    # Ensure proper spacing around fractions
    text = re.sub(r'\\frac\{(.*?)\}\{(.*?)\}', r'\\frac{\1}{\2}', text)
    
    # Fix double dollar signs
    text = re.sub(r'\$\$\s*\$\$', r'$$', text)
    
    return text
'''
        
        # Replace the original function with the enhanced one
        modified_content = content.replace(original_function, enhanced_function)
        
        # Write back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print("Successfully enhanced fix_latex_formatting function")
    else:
        print("Could not find fix_latex_formatting function")

if __name__ == "__main__":
    fix_latex_format_function("math_agent.py") 