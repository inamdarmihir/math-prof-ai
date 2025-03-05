#!/usr/bin/env python3
"""
Comprehensive code review and fix for math_agent.py.
This script thoroughly examines the file structure and fixes all syntax issues.
"""
import os
import re

def comprehensive_fix(file_path):
    """
    Performs a comprehensive code review and fixes syntax issues line by line.
    
    Args:
        file_path (str): Path to the math_agent.py file
    """
    # Create a backup
    backup_file = f"{file_path}.comprehensive_backup"
    if not os.path.exists(backup_file):
        with open(file_path, 'r', encoding='utf-8') as src:
            with open(backup_file, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        print(f"Created backup at {backup_file}")
    
    # Read the file and identify potential issues
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for all instances of problematic docstrings
    docstring_pattern = r'(\s*)(Fix common LaTeX formatting issues.*?str: Text with properly formatted LaTeX)'
    docstrings = list(re.finditer(docstring_pattern, content, re.DOTALL))
    
    print(f"Found {len(docstrings)} instances of problematic docstring text")
    for i, match in enumerate(docstrings):
        start_pos = match.start()
        end_pos = match.end()
        
        # Get line numbers for the docstring
        line_start = content[:start_pos].count('\n') + 1
        line_end = content[:end_pos].count('\n') + 1
        
        print(f"Docstring {i+1}: Lines {line_start}-{line_end}")
        
        # Check if this is a properly formatted docstring or a floating one
        lines_before = content[:start_pos].split('\n')[-3:]
        is_floating = not any('def ' in line for line in lines_before)
        
        if is_floating:
            print(f"  This appears to be a floating docstring")
        else:
            print(f"  This appears to be part of a function definition")
    
    # Create a clean version of the content
    lines = content.split('\n')
    clean_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this is the start of a problematic docstring
        if "Fix common LaTeX formatting issues" in line and i > 0:
            prev_line = lines[i-1].strip()
            
            # If the previous line is not part of a function definition, this is a floating docstring
            if not prev_line.startswith('def ') and not prev_line.startswith('"""') and not prev_line.startswith("'''"):
                print(f"Skipping floating docstring starting at line {i+1}")
                
                # Skip until we find a blank line or the end of the docstring section
                while i < len(lines) and not lines[i].strip() == "":
                    i += 1
                    if i < len(lines) and "properly formatted LaTeX" in lines[i]:
                        i += 2  # Skip one more line after the end of the docstring
                        break
            else:
                clean_lines.append(line)
        else:
            clean_lines.append(line)
        
        i += 1
    
    # Write the cleaned content back to the file
    clean_content = '\n'.join(clean_lines)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(clean_content)
    
    print("\nRemoved all floating docstrings")
    
    # Verify fix_latex_formatting function
    with open(file_path, 'r', encoding='utf-8') as f:
        updated_content = f.read()
    
    if "def fix_latex_formatting" in updated_content:
        print("\nVerifying fix_latex_formatting function:")
        
        # Extract the function
        func_match = re.search(r'def fix_latex_formatting.*?return text', updated_content, re.DOTALL)
        if func_match:
            func_text = func_match.group(0)
            func_start_line = updated_content[:func_match.start()].count('\n') + 1
            
            print(f"Function starts at line {func_start_line}")
            
            # Check if function has a proper docstring
            docstring_match = re.search(r'def fix_latex_formatting.*?\s+""".*?"""', func_text, re.DOTALL)
            if not docstring_match:
                print("Function is missing a proper docstring, adding one")
                
                # Add proper docstring
                func_def_line = re.search(r'def fix_latex_formatting.*?:', func_text).group(0)
                docstring = '''    """Fix common LaTeX formatting issues in the Math Agent output.
    
    This function addresses several common formatting problems:
    1. Removes unnecessary square brackets around align environments
    2. Fixes backslash spacing issues
    3. Ensures proper delimiters for block and inline equations
    
    Args:
        text (str): The text containing LaTeX equations
        
    Returns:
        str: Text with properly formatted LaTeX
    """'''
                
                new_func_text = func_text.replace(func_def_line, func_def_line + "\n" + docstring)
                updated_content = updated_content.replace(func_text, new_func_text)
                
                # Write the updated content back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                
                print("Added docstring to function")
    
    # Now check if the output_guardrails function correctly uses fix_latex_formatting
    output_guardrails_pattern = r'def output_guardrails\(.*?\).*?return.*?'
    output_match = re.search(output_guardrails_pattern, updated_content, re.DOTALL)
    
    if output_match:
        output_func = output_match.group(0)
        
        # Check if fix_latex_formatting is called in the function
        if "fix_latex_formatting" not in output_func:
            print("\noutput_guardrails function doesn't call fix_latex_formatting, adding call")
            
            # Find the return statement
            return_match = re.search(r'(\s+)return(\s+)([a-zA-Z_]+)', output_func)
            if return_match:
                indent = return_match.group(1)
                return_var = return_match.group(3)
                
                # Add the call before the return
                modified_return = f"{indent}{return_var} = fix_latex_formatting({return_var}){indent}return{return_match.group(2)}{return_var}"
                modified_output_func = output_func.replace(return_match.group(0), modified_return)
                
                # Update the file
                updated_content = updated_content.replace(output_func, modified_output_func)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                
                print(f"Added fix_latex_formatting call before returning {return_var}")
    
    print("\nFix attempt completed. Please run 'streamlit run math_agent.py' to verify the fix.")

if __name__ == "__main__":
    comprehensive_fix("math_agent.py") 