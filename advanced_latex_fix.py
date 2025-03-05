#!/usr/bin/env python3
import re
import os
import shutil

def enhance_fix_latex_formatting(file_path):
    """
    Enhance the fix_latex_formatting function in math_agent.py to better handle LaTeX formatting issues.
    
    Args:
        file_path (str): Path to the math_agent.py file
    """
    # Create a backup
    backup_file = f"{file_path}.latex_backup"
    shutil.copy2(file_path, backup_file)
    print(f"Backup created at {backup_file}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Regular expression to find the existing fix_latex_formatting function
    function_pattern = re.compile(r'def fix_latex_formatting\(text\):.*?(?=def|\Z)', re.DOTALL)
    
    # Define the new enhanced function
    new_function = '''def fix_latex_formatting(text):
    """
    Fix common LaTeX formatting issues in text.
    
    This function addresses various LaTeX formatting problems, including:
    - Handling non-string inputs
    - Fixing align environment formatting issues
    - Correcting nested delimiters ($$, \\[, etc.)
    - Ensuring proper spacing and line breaks
    - Handling missing environment delimiters
    
    Args:
        text: Input text that may contain LaTeX expressions
        
    Returns:
        str: Text with fixed LaTeX formatting
    """
    # Handle non-string inputs
    if not isinstance(text, str):
        if isinstance(text, list):
            # For lists, apply the function to each element
            return [fix_latex_formatting(item) for item in text]
        # Convert other types to string
        text = str(text)
    
    # Fix missing or malformed align environments
    # First, detect align environments without proper begin/end tags
    text = re.sub(r'(?<!\\\begin{align})(\s*[\d\w]+\s*[\+\-\*\/]?\s*[\d\w]+\s*&\s*[=<>].*?)(?!\\\end{align})',
                 r'\\begin{align}\n\\1\n\\end{align}', text)
    
    # Fix improper align environment with square brackets
    text = re.sub(r'\[\s*\\begin\{align\*?\}(.*?)\\end\{align\*?\}\s*\]',
                 r'\\begin{align}\1\\end{align}', text, flags=re.DOTALL)
    
    # Fix nested delimiters ($$...$$ inside \\[...\\] or vice versa)
    text = re.sub(r'\\\[\s*\$\$(.*?)\$\$\s*\\\]', r'\\[\1\\]', text, flags=re.DOTALL)
    text = re.sub(r'\$\$\s*\\\[(.*?)\\\]\s*\$\$', r'$$\1$$', text, flags=re.DOTALL)
    
    # Fix missing end of align environments
    text = re.sub(r'\\begin\{align\}(.*?)(?!\\end\{align\})(?=\\begin|$)', 
                 r'\\begin{align}\1\\end{align}', text, flags=re.DOTALL)
    
    # Properly format alignment operators in align environments
    text = re.sub(r'(\\begin\{align(?:\*?)\}.*?)(\\\\)', r'\1\\\\\\\\', text, flags=re.DOTALL)
    text = re.sub(r'(\\begin\{align(?:\*?)\}.*?)([^&])=', r'\1\2&=', text, flags=re.DOTALL)
    
    # Fix spacing after line breaks in align environments
    text = re.sub(r'\\\\\\\\(\s*)', r'\\\\\\\\ \n', text)
    
    # Remove extraneous closing brackets after align environments
    text = re.sub(r'\\end\{align\*?\}\s*\]', r'\\end{align}', text)
    
    # Ensure proper equation delimiters
    # If there's standalone LaTeX without delimiters, wrap it in $$
    text = re.sub(r'(?<!\$)(?<!\\\\)(?<!\\\[)(\\frac\{.*?\}\{.*?\}|\\int|\\sum|\\prod|\\lim|\\mathbb\{[A-Z]\})(?!\$)(?!\\\\)(?!\\\])',
                 r'$$\1$$', text)
    
    return text
'''
    
    # Replace the old function with the new one
    if function_pattern.search(content):
        new_content = function_pattern.sub(new_function, content)
        
        # Write the modified content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"LaTeX formatting function in {file_path} has been enhanced")
    else:
        print(f"Could not find the fix_latex_formatting function in {file_path}")

if __name__ == "__main__":
    enhance_fix_latex_formatting("math_agent.py")
    print("LaTeX formatting enhancement completed") 