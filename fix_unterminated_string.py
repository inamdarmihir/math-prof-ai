#!/usr/bin/env python3
"""
Fix for unterminated string literal in the LaTeX formatting function
"""
import os
import re

def fix_unterminated_string(file_path):
    """
    Fixes the unterminated string literal in the fix_latex_formatting function
    
    Args:
        file_path (str): Path to the math_agent.py file
    """
    # Create backup
    backup_file = f"{file_path}.string_fix"
    if not os.path.exists(backup_file):
        with open(file_path, 'r', encoding='utf-8') as src:
            with open(backup_file, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        print(f"Created backup at {backup_file}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find and fix the problematic line
    for i, line in enumerate(lines):
        if r"text = re.sub(r'\\\\(\s+)', r'\\\\ \n" in line:
            # Fix the unterminated string literal
            lines[i] = "    text = re.sub(r'\\\\\\\\(\\s+)', r'\\\\\\\\ ' + '\\n', text)\n"
            print(f"Fixed unterminated string literal on line {i+1}")
    
    # Write the fixed content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("Fixed string literal issue in fix_latex_formatting function")

if __name__ == "__main__":
    fix_unterminated_string("math_agent.py") 