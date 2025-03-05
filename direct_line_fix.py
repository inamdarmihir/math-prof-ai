#!/usr/bin/env python3
"""
Direct line fix for math_agent.py - specifically removes lines 68-80 containing the floating docstring.
"""
import os

def fix_specific_lines(file_path):
    """
    Directly remove lines 68-80 which contain the floating docstring.
    
    Args:
        file_path (str): Path to the math_agent.py file
    """
    # Create a backup if one doesn't already exist
    backup_file = f"{file_path}.direct_backup"
    if not os.path.exists(backup_file):
        with open(file_path, 'r', encoding='utf-8') as src:
            with open(backup_file, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        print(f"Created backup at {backup_file}")
    
    # Read the file line by line
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Print information about line 68 and surrounding lines
    start_line = max(0, 67-2)
    end_line = min(len(lines), 80+2)
    print(f"\nOriginal content around line 68-80:")
    for i in range(start_line, end_line):
        print(f"Line {i+1}: {lines[i].rstrip()}")
    
    # Directly remove lines 68-80 (0-based index is 67-79)
    new_lines = lines[:67] + lines[80:]
    
    # Write the fixed content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"\nRemoved lines 68-80 (floating docstring)")
    
    # Display the modified content
    with open(file_path, 'r', encoding='utf-8') as f:
        updated_lines = f.readlines()
    
    new_start_line = max(0, 67-2)
    new_end_line = min(len(updated_lines), 67+2)
    print("\nContent after removing floating docstring:")
    for i in range(new_start_line, new_end_line):
        print(f"Line {i+1}: {updated_lines[i].rstrip()}")
    
    print("\nFix attempt completed. Please run 'streamlit run math_agent.py' to verify the fix.")

if __name__ == "__main__":
    fix_specific_lines("math_agent.py") 