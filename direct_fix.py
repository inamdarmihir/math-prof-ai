import re

def direct_fix(file_path):
    """
    Directly fix the indentation error by editing the file content.
    
    Args:
        file_path (str): Path to the math_agent.py file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find the problematic lines (approximate location)
        problem_start = 0
        for i, line in enumerate(lines):
            if "def fix_latex_formatting" in line:
                problem_start = i
                break
        
        if problem_start == 0:
            print("Could not find the function definition")
            return
        
        # Print the problematic lines for debugging
        print(f"Found function at line {problem_start+1}")
        for i in range(problem_start, min(problem_start+10, len(lines))):
            print(f"Line {i+1}: {lines[i].rstrip()}")
        
        # Create a new version of the file content
        new_lines = []
        skip_mode = False
        
        for i, line in enumerate(lines):
            # If we're at the function definition, enter skip mode
            if "def fix_latex_formatting" in line and not skip_mode:
                skip_mode = True
                # Add our fixed version instead
                new_lines.append("def fix_latex_formatting(text):\n")
                new_lines.append("    \"\"\"Fix common LaTeX formatting issues in the Math Agent output.\n")
                new_lines.append("    \n")
                new_lines.append("    This function addresses several common formatting problems:\n")
                new_lines.append("    1. Removes unnecessary square brackets around align environments\n")
                new_lines.append("    2. Fixes backslash spacing issues\n")
                new_lines.append("    3. Ensures proper delimiters for block and inline equations\n")
                new_lines.append("    \n")
                new_lines.append("    Args:\n")
                new_lines.append("        text (str): The text containing LaTeX equations\n")
                new_lines.append("    \n")
                new_lines.append("    Returns:\n")
                new_lines.append("        str: Text with properly formatted LaTeX\n")
                new_lines.append("    \"\"\"\n")
                new_lines.append("    # Fix align environments with square brackets\n")
                new_lines.append("    text = re.sub(r'\\[\\s*\\\\begin\\{align\\*?\\}(.*?)\\\\end\\{align\\*?\\}\\s*\\]',\n")
                new_lines.append("                 r'$$\\\\begin{align}\\1\\\\end{align}$$',\n")
                new_lines.append("                 text, flags=re.DOTALL)\n")
                new_lines.append("    \n")
                new_lines.append("    # Fix standalone align environments (without brackets)\n")
                new_lines.append("    text = re.sub(r'\\\\begin\\{align\\*?\\}(.*?)\\\\end\\{align\\*?\\}',\n")
                new_lines.append("                 r'$$\\\\begin{align}\\1\\\\end{align}$$',\n")
                new_lines.append("                 text, flags=re.DOTALL)\n")
                new_lines.append("    \n")
                new_lines.append("    # Fix incorrect line breaks in align environments\n")
                new_lines.append("    text = re.sub(r'\\\\\\\\(\\s+)', r'\\\\\\\\ \\n', text)\n")
                new_lines.append("    \n")
                new_lines.append("    # Fix &= spacing issues\n")
                new_lines.append("    text = re.sub(r'&=\\s+\\\\', r'&= \\\\', text)\n")
                new_lines.append("    \n")
                new_lines.append("    # Ensure single variables are properly formatted with inline math\n")
                new_lines.append("    text = re.sub(r'(?<![\\\\$a-zA-Z0-9])\\b([a-zA-Z])\\b(?![\\\\$a-zA-Z0-9=])', r'$\\1$', text)\n")
                new_lines.append("    \n")
                new_lines.append("    # Fix consecutive equation blocks (ensure proper spacing)\n")
                new_lines.append("    text = re.sub(r'\\$\\$\\s*\\$\\$', r'$$\\n\\n$$', text)\n")
                new_lines.append("    \n")
                new_lines.append("    return text\n")
                continue
            
            # If we're in skip mode, look for the end of the function
            if skip_mode and line.strip() == "return text":
                skip_mode = False
                continue
            
            # If we're in skip mode, skip this line
            if skip_mode:
                continue
            
            # Otherwise, add the line to our new content
            new_lines.append(line)
        
        # Write the new content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        print(f"Successfully fixed {file_path}")
    except Exception as e:
        print(f"Error fixing file: {str(e)}")

if __name__ == "__main__":
    file_path = "math_agent.py"
    direct_fix(file_path) 