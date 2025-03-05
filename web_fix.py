#!/usr/bin/env python3
"""
Fix indentation errors in math_agent.py by completely rebuilding the problematic sections.
This script applies recommended Python indentation practices to resolve IndentationError.
"""
import re
import os
import sys

def rebuild_math_agent(file_path):
    """
    Completely rebuild the math_agent.py file to fix indentation issues.
    
    Args:
        file_path (str): Path to the math_agent.py file
    """
    # Create a backup of the original file
    backup_file = f"{file_path}.indentation_backup"
    if not os.path.exists(backup_file):
        try:
            with open(file_path, 'r', encoding='utf-8') as src:
                with open(backup_file, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            print(f"Created backup at {backup_file}")
        except Exception as e:
            print(f"Error creating backup: {e}")
            return
    
    try:
        # Read the entire file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Step 1: Remove the problematic fix_latex_formatting function if it exists
        latex_func_pattern = r'def fix_latex_formatting[\s\S]*?return text\s*\n'
        content = re.sub(latex_func_pattern, '', content)
        
        # Step 2: Insert a properly indented version after imports
        import_section_end = content.find("\n\n", content.find("import"))
        if import_section_end == -1:
            import_section_end = content.find("\n", content.rfind("import"))
        
        # Define the properly indented function with triple quotes properly escaped
        latex_formatting_function = '''
def fix_latex_formatting(text):
    """Fix common LaTeX formatting issues in the Math Agent output.
    
    This function addresses several common formatting problems:
    1. Removes unnecessary square brackets around align environments
    2. Fixes backslash spacing issues
    3. Ensures proper delimiters for block and inline equations
    
    Args:
        text (str): The text containing LaTeX equations
        
    Returns:
        str: Text with properly formatted LaTeX
    """
    # Fix align environments with square brackets
    text = re.sub(r'\\[\\s*\\\\begin\\{align\\*?\\}(.*?)\\\\end\\{align\\*?\\}\\s*\\]', 
                 r'$$\\\\begin{align}\\1\\\\end{align}$$', 
                 text, flags=re.DOTALL)
    
    # Fix standalone align environments (without brackets)
    text = re.sub(r'\\\\begin\\{align\\*?\\}(.*?)\\\\end\\{align\\*?\\}', 
                 r'$$\\\\begin{align}\\1\\\\end{align}$$', 
                 text, flags=re.DOTALL)
    
    # Fix incorrect line breaks in align environments
    text = re.sub(r'\\\\\\\\(\\s+)', r'\\\\\\\\ \\n', text)
    
    # Fix &= spacing issues
    text = re.sub(r'&=\\s+\\\\', r'&= \\\\', text)
    
    # Ensure single variables are properly formatted with inline math
    text = re.sub(r'(?<![\\\\$a-zA-Z0-9])\\b([a-zA-Z])\\b(?![\\\\$a-zA-Z0-9=])', r'$\\1$', text)
    
    # Fix consecutive equation blocks (ensure proper spacing)
    text = re.sub(r'\\$\\$\\s*\\$\\$', r'$$\\n\\n$$', text)
    
    return text
'''
        
        # Insert the function after imports
        if import_section_end > 0:
            content = content[:import_section_end + 2] + latex_formatting_function + content[import_section_end + 2:]
        else:
            print("Warning: Could not find imports section, adding function at the beginning")
            content = latex_formatting_function + content
        
        # Step 3: Update output_guardrails function to use fix_latex_formatting
        if "def output_guardrails" in content:
            # Pattern to match the return statement in output_guardrails
            return_pattern = r'(\s+)return(\s+)([a-zA-Z_]+)(\s*\n)'
            
            # Find all matches to handle potential multiple occurrences
            matches = list(re.finditer(return_pattern, content))
            
            # Process from the last match to avoid offset issues
            for match in reversed(matches):
                # Check if it's not already modified
                pre_match_text = content[max(0, match.start() - 50):match.start()]
                if "fix_latex_formatting" not in pre_match_text:
                    # Extract indentation and variable name
                    indent = match.group(1)
                    var_name = match.group(3)
                    
                    # Replace the return statement
                    replacement = f"{indent}{var_name} = fix_latex_formatting({var_name}){indent}return{match.group(2)}{var_name}{match.group(4)}"
                    content = content[:match.start()] + replacement + content[match.end():]
        
        # Write the modified content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Successfully fixed indentation issues in {file_path}")
        print("The script has implemented the following fixes:")
        print("1. Removed the problematic fix_latex_formatting function")
        print("2. Added a properly indented version of the function")
        print("3. Updated the output_guardrails function to use fix_latex_formatting")
        print("\nPlease run 'streamlit run math_agent.py' to verify the fix.")
        
    except Exception as e:
        print(f"Error fixing file: {str(e)}")
        print("Traceback:", sys.exc_info())

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "math_agent.py"
    
    rebuild_math_agent(file_path) 