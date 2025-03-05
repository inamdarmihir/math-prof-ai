#!/usr/bin/env python3
"""
Focused fix for math_agent.py docstring syntax error on line 68.
"""
import os

def fix_docstring_syntax(file_path):
    """
    Fix the docstring syntax error in math_agent.py.
    
    Args:
        file_path (str): Path to the math_agent.py file
    """
    # Create a backup if one doesn't already exist
    backup_file = f"{file_path}.backup_syntax"
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
    end_line = min(len(lines), 68+5)
    print(f"\nOriginal content around line 68:")
    for i in range(start_line, end_line):
        print(f"Line {i+1}: {lines[i].rstrip()}")
    
    # Try to identify docstring pattern and fix it
    docstring_start = 0
    for i, line in enumerate(lines):
        if "def fix_latex_formatting" in line:
            docstring_start = i + 1
            break
    
    print(f"\nIdentified potential function definition at line {docstring_start}")
    
    # Fix the docstring if found
    if docstring_start > 0 and docstring_start < len(lines):
        # Check if the next line contains a docstring
        if '"""' in lines[docstring_start]:
            # Replace the problematic docstring definition
            new_lines = lines.copy()
            
            # Keep track of the function definition line
            func_def_line = new_lines[docstring_start-1]
            
            # Create a proper docstring
            new_lines[docstring_start] = '    """Fix common LaTeX formatting issues in the Math Agent output.\n'
            
            # Check if there are more docstring lines
            current_line = docstring_start + 1
            while current_line < len(new_lines) and '"""' not in new_lines[current_line]:
                current_line += 1
            
            # If we found the closing quotes, make sure they're properly formatted
            if current_line < len(new_lines) and '"""' in new_lines[current_line]:
                new_lines[current_line] = '    """\n'
            
            # Write the fixed content back to the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            
            print(f"\nFixed docstring format at line {docstring_start}")
            print("Modified content around line 68:")
            
            # Display the modifications
            with open(file_path, 'r', encoding='utf-8') as f:
                updated_lines = f.readlines()
            
            start_line = max(0, 68-5)
            end_line = min(len(updated_lines), 68+5)
            for i in range(start_line, end_line):
                print(f"Line {i+1}: {updated_lines[i].rstrip()}")
        else:
            print("The line after function definition doesn't appear to be a docstring")
            
            # More aggressive approach: completely replace the function
            print("\nAttempting more aggressive fix by rebuilding the function")
            
            # Find the full function to replace
            func_start = docstring_start - 1
            func_end = func_start + 1
            
            # Find where the function ends (first return statement)
            while func_end < len(lines) and "return text" not in lines[func_end]:
                func_end += 1
            
            if func_end < len(lines):
                func_end += 1  # Include the return line
                
                # Create the replacement function with proper docstring
                replacement = [
                    "def fix_latex_formatting(text):\n",
                    "    \"\"\"Fix common LaTeX formatting issues in the Math Agent output.\n",
                    "    \n",
                    "    This function addresses several common formatting problems:\n",
                    "    1. Removes unnecessary square brackets around align environments\n",
                    "    2. Fixes backslash spacing issues\n",
                    "    3. Ensures proper delimiters for block and inline equations\n",
                    "    \n",
                    "    Args:\n",
                    "        text (str): The text containing LaTeX equations\n",
                    "        \n",
                    "    Returns:\n",
                    "        str: Text with properly formatted LaTeX\n",
                    "    \"\"\"\n",
                    "    # Fix align environments with square brackets\n",
                    "    text = re.sub(r'\\[\\s*\\\\begin\\{align\\*?\\}(.*?)\\\\end\\{align\\*?\\}\\s*\\]',\n",
                    "                 r'$$\\\\begin{align}\\1\\\\end{align}$$',\n",
                    "                 text, flags=re.DOTALL)\n",
                    "    \n",
                    "    # Fix standalone align environments (without brackets)\n",
                    "    text = re.sub(r'\\\\begin\\{align\\*?\\}(.*?)\\\\end\\{align\\*?\\}',\n",
                    "                 r'$$\\\\begin{align}\\1\\\\end{align}$$',\n",
                    "                 text, flags=re.DOTALL)\n",
                    "    \n",
                    "    # Fix incorrect line breaks in align environments\n",
                    "    text = re.sub(r'\\\\\\\\(\\s+)', r'\\\\\\\\ \\n', text)\n",
                    "    \n",
                    "    # Fix &= spacing issues\n",
                    "    text = re.sub(r'&=\\s+\\\\', r'&= \\\\', text)\n",
                    "    \n",
                    "    # Ensure single variables are properly formatted with inline math\n",
                    "    text = re.sub(r'(?<![\\\\$a-zA-Z0-9])\\b([a-zA-Z])\\b(?![\\\\$a-zA-Z0-9=])', r'$\\1$', text)\n",
                    "    \n",
                    "    # Fix consecutive equation blocks (ensure proper spacing)\n",
                    "    text = re.sub(r'\\$\\$\\s*\\$\\$', r'$$\\n\\n$$', text)\n",
                    "    \n",
                    "    return text\n"
                ]
                
                # Replace the function with our new version
                new_lines = lines[:func_start] + replacement + lines[func_end:]
                
                # Write the fixed content back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                
                print(f"\nReplaced the entire function from line {func_start+1} to {func_end}")
                
                # Display the modifications
                with open(file_path, 'r', encoding='utf-8') as f:
                    updated_lines = f.readlines()
                
                start_line = max(0, func_start-2)
                end_line = min(len(updated_lines), func_start + len(replacement) + 2)
                for i in range(start_line, end_line):
                    print(f"Line {i+1}: {updated_lines[i].rstrip()}")
            else:
                print("Could not find the end of the function to replace")
    else:
        print("Could not find the fix_latex_formatting function definition")
    
    print("\nFix attempt completed. Please run 'streamlit run math_agent.py' to verify the fix.")

if __name__ == "__main__":
    fix_docstring_syntax("math_agent.py") 