#!/usr/bin/env python3
"""
Fix for unterminated string literal on line 405 of math_agent.py
"""
import os

def fix_unterminated_string(file_path):
    """
    Fixes the unterminated string literal on line 405.
    
    Args:
        file_path (str): Path to the math_agent.py file
    """
    # Create a backup
    backup_file = f"{file_path}.string_backup"
    if not os.path.exists(backup_file):
        with open(file_path, 'r', encoding='utf-8') as src:
            with open(backup_file, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        print(f"Created backup at {backup_file}")
    
    # Read the file line by line
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Print the problematic region
    start_line = max(400, 0)
    end_line = min(410, len(lines))
    
    print("\nOriginal content around line 405:")
    for i in range(start_line, end_line):
        print(f"{i+1}: {lines[i].rstrip()}")
    
    # Identify and fix the unterminated string
    if len(lines) >= 405:  # Make sure line 405 exists
        problematic_line = lines[404]  # Line 405 (0-indexed is 404)
        
        # Check for unpaired quotes
        single_quotes = problematic_line.count("'")
        double_quotes = problematic_line.count('"')
        
        # If we have an odd number of quotes, it likely means an unclosed string
        if single_quotes % 2 != 0:
            # Fix: add a closing single quote at the end of the line if it doesn't already end with a quote
            if not problematic_line.rstrip().endswith("'"):
                lines[404] = problematic_line.rstrip() + "'\n"
                print("\nFixed by adding closing single quote")
        elif double_quotes % 2 != 0:
            # Fix: add a closing double quote at the end of the line if it doesn't already end with a quote
            if not problematic_line.rstrip().endswith('"'):
                lines[404] = problematic_line.rstrip() + '"\n'
                print("\nFixed by adding closing double quote")
        else:
            # If quotes are balanced, the issue might be on the next line or a triple quote issue
            # Check if this is the beginning of a docstring (triple quotes)
            if '"""' in problematic_line and problematic_line.count('"""') % 2 != 0:
                # This is an unclosed triple quote docstring
                lines[404] = problematic_line.rstrip() + '"""\n'
                print("\nFixed by adding closing triple double quotes")
            elif "'''" in problematic_line and problematic_line.count("'''") % 2 != 0:
                # This is an unclosed triple quote docstring
                lines[404] = problematic_line.rstrip() + "'''\n"
                print("\nFixed by adding closing triple single quotes")
    
    # Write the fixed content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("\nAfter fixing:")
    # Read the file again to show the fixed content
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i in range(start_line, end_line):
        print(f"{i+1}: {lines[i].rstrip()}")
    
    print("\nFix attempt completed. Please run 'streamlit run math_agent.py' to verify the fix.")

if __name__ == "__main__":
    fix_unterminated_string("math_agent.py") 