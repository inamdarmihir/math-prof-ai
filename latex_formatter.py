import re

def fix_latex_formatting(text):
    """
    Fix common LaTeX formatting issues in the Math Agent output.
    
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
    text = re.sub(r'\[\s*\\begin\{align\*?\}(.*?)\\end\{align\*?\}\s*\]', 
                 r'$$\\begin{align}\1\\end{align}$$', 
                 text, flags=re.DOTALL)
    
    # Fix standalone align environments (without brackets)
    text = re.sub(r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}', 
                 r'$$\\begin{align}\1\\end{align}$$', 
                 text, flags=re.DOTALL)
    
    # Fix incorrect line breaks in align environments
    text = re.sub(r'\\\\(\s+)', r'\\\\ \n', text)
    
    # Fix &= spacing issues
    text = re.sub(r'&=\s+\\', r'&= \\', text)
    
    # Ensure single variables are properly formatted with inline math
    # This regex looks for single letters that should be math variables but aren't in math mode
    text = re.sub(r'(?<![\\$a-zA-Z0-9])\b([a-zA-Z])\b(?![\\$a-zA-Z0-9=])', r'$\1$', text)
    
    # Fix consecutive equation blocks (ensure proper spacing)
    text = re.sub(r'\$\$\s*\$\$', r'$$\n\n$$', text)
    
    return text

def apply_latex_fixes_to_file(file_path):
    """
    Apply LaTeX formatting fixes to a file.
    
    Args:
        file_path (str): Path to the file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        fixed_content = fix_latex_formatting(content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
            
        print(f"Successfully fixed LaTeX formatting in {file_path}")
    except Exception as e:
        print(f"Error fixing LaTeX formatting: {str(e)}")

def modify_output_guardrails(math_agent_path):
    """
    Modify the output_guardrails function in math_agent.py to apply LaTeX fixes
    
    Args:
        math_agent_path (str): Path to math_agent.py
    """
    try:
        with open(math_agent_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for the output_guardrails function
        guardrails_pattern = r'def output_guardrails\(answer\):.*?return\s+[a-zA-Z_]+\s*\n'
        guardrails_match = re.search(guardrails_pattern, content, flags=re.DOTALL)
        
        if not guardrails_match:
            print("Could not find output_guardrails function in the file")
            return
        
        original_function = guardrails_match.group(0)
        
        # Modify the function to apply LaTeX fixes
        modified_function = original_function.replace(
            "return modified_answer", 
            "modified_answer = fix_latex_formatting(modified_answer)\n    return modified_answer"
        )
        
        if "def fix_latex_formatting" not in content:
            # Add the fix_latex_formatting function
            import_section_end = content.find("\n\n", content.find("import"))
            if import_section_end == -1:
                import_section_end = content.find("\n", content.rfind("import"))
            
            content = content[:import_section_end + 2] + \
                     fix_latex_formatting.__doc__ + "\n" + \
                     "def fix_latex_formatting(text):\n" + \
                     "    # Fix align environments with square brackets\n" + \
                     "    text = re.sub(r'\\[\\s*\\\\begin\\{align\\*?\\}(.*?)\\\\end\\{align\\*?\\}\\s*\\]', \n" + \
                     "                 r'$$\\\\begin{align}\\1\\\\end{align}$$', \n" + \
                     "                 text, flags=re.DOTALL)\n" + \
                     "    \n" + \
                     "    # Fix standalone align environments (without brackets)\n" + \
                     "    text = re.sub(r'\\\\begin\\{align\\*?\\}(.*?)\\\\end\\{align\\*?\\}', \n" + \
                     "                 r'$$\\\\begin{align}\\1\\\\end{align}$$', \n" + \
                     "                 text, flags=re.DOTALL)\n" + \
                     "    \n" + \
                     "    # Fix incorrect line breaks in align environments\n" + \
                     "    text = re.sub(r'\\\\\\\\(\\s+)', r'\\\\\\\\ \\n', text)\n" + \
                     "    \n" + \
                     "    # Fix &= spacing issues\n" + \
                     "    text = re.sub(r'&=\\s+\\\\', r'&= \\\\', text)\n" + \
                     "    \n" + \
                     "    # Ensure single variables are properly formatted with inline math\n" + \
                     "    text = re.sub(r'(?<![\\\\$a-zA-Z0-9])\\b([a-zA-Z])\\b(?![\\\\$a-zA-Z0-9=])', r'$\\1$', text)\n" + \
                     "    \n" + \
                     "    return text\n\n" + \
                     content[import_section_end + 2:]
        
        # Replace the original function with the modified one
        content = content.replace(original_function, modified_function)
        
        with open(math_agent_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"Successfully modified output_guardrails in {math_agent_path}")
    except Exception as e:
        print(f"Error modifying output_guardrails: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        math_agent_path = sys.argv[1]
    else:
        math_agent_path = "math_agent.py"
    
    modify_output_guardrails(math_agent_path) 