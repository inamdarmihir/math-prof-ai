import re

def fix_indentation_error(file_path):
    """
    Fix the indentation error in the LaTeX formatter function docstring.
    
    Args:
        file_path (str): Path to the math_agent.py file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Alternative approach: completely replace the function
        print("Using direct replacement approach to fix the function")
        
        # First, find where to insert the function
        import_section_end = content.find("\n\n", content.find("import"))
        if import_section_end == -1:
            import_section_end = content.find("\n", content.rfind("import"))
        
        # Define the fixed function with proper indentation
        fixed_func = """
def fix_latex_formatting(text):
    \"\"\"Fix common LaTeX formatting issues in the Math Agent output.
    
    This function addresses several common formatting problems:
    1. Removes unnecessary square brackets around align environments
    2. Fixes backslash spacing issues
    3. Ensures proper delimiters for block and inline equations
    
    Args:
        text (str): The text containing LaTeX equations
        
    Returns:
        str: Text with properly formatted LaTeX
    \"\"\"
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
"""
        
        # Check if the function already exists
        if "def fix_latex_formatting" in content:
            # Find and replace the existing function
            func_pattern = r'def fix_latex_formatting\(text\):[\s\S]+?return text'
            func_match = re.search(func_pattern, content)
            
            if func_match:
                original_func = func_match.group(0)
                content = content.replace(original_func, fixed_func.strip())
            else:
                print("Found function declaration but couldn't match the full function")
        else:
            # Add the function after imports
            content = content[:import_section_end + 2] + fixed_func + content[import_section_end + 2:]
        
        # Modify the output_guardrails function if it exists
        if "def output_guardrails" in content:
            guardrails_pattern = r'def output_guardrails\(answer\):[\s\S]+?return\s+[a-zA-Z_]+\s*'
            guardrails_match = re.search(guardrails_pattern, content)
            
            if guardrails_match:
                original_function = guardrails_match.group(0)
                
                # Check if the fix is already applied
                if "fix_latex_formatting" not in original_function:
                    modified_function = original_function.replace(
                        "return modified_answer", 
                        "modified_answer = fix_latex_formatting(modified_answer)\n    return modified_answer"
                    )
                    content = content.replace(original_function, modified_function)
                    print("Modified output_guardrails function to apply LaTeX fixes")
                else:
                    print("output_guardrails function already applies LaTeX fixes")
            else:
                print("Could not match the full output_guardrails function")
        else:
            print("output_guardrails function not found")
            
        # Write the updated content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Successfully fixed indentation error in {file_path}")
    except Exception as e:
        print(f"Error fixing indentation: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        math_agent_path = sys.argv[1]
    else:
        math_agent_path = "math_agent.py"
    
    fix_indentation_error(math_agent_path) 