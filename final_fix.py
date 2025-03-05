#!/usr/bin/env python3
"""
Final fix for math_agent.py - remove floating docstring at line 68.
"""
import os

def fix_floating_docstring(file_path):
    """
    Fix the floating docstring syntax error in math_agent.py.
    
    Args:
        file_path (str): Path to the math_agent.py file
    """
    # Create a backup if one doesn't already exist
    backup_file = f"{file_path}.final_backup"
    if not os.path.exists(backup_file):
        with open(file_path, 'r', encoding='utf-8') as src:
            with open(backup_file, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        print(f"Created backup at {backup_file}")
    
    # Read the file line by line
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Print information about line 68 and surrounding lines
    start_line = max(0, 68-5)
    end_line = min(len(lines), 68+15)
    print(f"\nOriginal content around line 68:")
    for i in range(start_line, end_line):
        print(f"Line {i+1}: {lines[i].rstrip()}")
    
    # Identify the floating docstring section
    floating_start = 0
    floating_end = 0
    
    # Let's find the line "Fix common LaTeX formatting issues" and the corresponding block
    for i, line in enumerate(lines):
        if "Fix common LaTeX formatting issues" in line:
            floating_start = i
            # Find where this section ends (look for blank line)
            for j in range(i+1, len(lines)):
                if lines[j].strip() == "":
                    floating_end = j
                    break
            break
    
    print(f"\nIdentified floating docstring from line {floating_start+1} to {floating_end+1}")
    
    # Remove the floating docstring
    if floating_start > 0:
        new_lines = lines[:floating_start] + lines[floating_end+1:]
        
        # Write the fixed content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        print(f"\nRemoved floating docstring (lines {floating_start+1} to {floating_end+1})")
        
        # Display the modified content
        with open(file_path, 'r', encoding='utf-8') as f:
            updated_lines = f.readlines()
        
        new_start_line = max(0, floating_start-5)
        new_end_line = min(len(updated_lines), floating_start+5)
        print("\nContent after removing floating docstring:")
        for i in range(new_start_line, new_end_line):
            print(f"Line {i+1}: {updated_lines[i].rstrip()}")
    else:
        print("Could not identify the floating docstring")
    
    # Now let's make sure the fix_latex_formatting function has a proper docstring
    func_line = 0
    for i, line in enumerate(updated_lines):
        if "def fix_latex_formatting" in line:
            func_line = i
            break
    
    if func_line > 0:
        print(f"\nFound fix_latex_formatting function at line {func_line+1}")
        
        # Check if it already has a proper docstring
        if func_line+1 < len(updated_lines) and '"""' in updated_lines[func_line+1]:
            print("Function already has a docstring, no further action needed")
        else:
            # Insert proper docstring
            docstring = [
                '    """Fix common LaTeX formatting issues in the Math Agent output.\n',
                '    \n',
                '    This function addresses several common formatting problems:\n',
                '    1. Removes unnecessary square brackets around align environments\n',
                '    2. Fixes backslash spacing issues\n',
                '    3. Ensures proper delimiters for block and inline equations\n',
                '    \n',
                '    Args:\n',
                '        text (str): The text containing LaTeX equations\n',
                '        \n',
                '    Returns:\n',
                '        str: Text with properly formatted LaTeX\n',
                '    """\n'
            ]
            
            new_lines = updated_lines[:func_line+1] + docstring + updated_lines[func_line+1:]
            
            # Write the fixed content back to the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            
            print(f"Added proper docstring to function")
    else:
        print("Could not find the fix_latex_formatting function")
    
    print("\nFix attempt completed. Please run 'streamlit run math_agent.py' to verify the fix.")

if __name__ == "__main__":
    fix_floating_docstring("math_agent.py") 