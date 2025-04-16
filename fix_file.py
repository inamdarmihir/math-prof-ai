#!/usr/bin/env python3

with open('math_agent_langgraph.py', 'r') as f:
    content = f.readlines()

# Find the position of the problematic code
found_except = False
start_line = 0
for i, line in enumerate(content):
    if 'except Exception as e:' in line:
        found_except = True
        start_line = i
    if found_except and 'def solve_derivative' in line:
        end_line = i
        break

if found_except:
    # Replace the problematic code
    new_content = content[:start_line+1]  # Keep up to the except line
    new_content.append('                    logging.error(f"Error processing quadratic equation: {str(e)}")\n')
    
    new_content.append('\n')
    new_content.append('    # If we reach this point, no equations were successfully solved\n')
    new_content.append('    logging.error("Could not solve any equations")\n')
    new_content.append('    if "result" not in state:\n')
    new_content.append('        state["result"] = {}\n')
    new_content.append('    state["result"]["error"] = "Could not solve the equations. Please check your input."\n')
    new_content.append('    state["execution_times"]["solve_equations"] = time.time() - start_time\n')
    new_content.append('    return state\n')
    new_content.append('\n')
    new_content.extend(content[end_line:])  # Add from def solve_derivative onwards
    
    # Write back the fixed content
    with open('math_agent_langgraph_fixed.py', 'w') as f:
        f.writelines(new_content)
    
    print("File fixed and saved as math_agent_langgraph_fixed.py")
else:
    print("Could not find the problematic code section") 