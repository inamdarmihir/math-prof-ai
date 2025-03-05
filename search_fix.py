#!/usr/bin/env python3
"""
Fix for TypeError in search_math_knowledge function
"""
import os
import re

def fix_search_function(file_path):
    """
    Fixes the TypeError in search_math_knowledge function
    
    Args:
        file_path (str): Path to the math_agent.py file
    """
    # Create backup
    backup_file = f"{file_path}.search_backup"
    if not os.path.exists(backup_file):
        with open(file_path, 'r', encoding='utf-8') as src:
            with open(backup_file, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        print(f"Created backup at {backup_file}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the search_math_knowledge function
    search_function_pattern = r'def search_math_knowledge\(query, limit=5\):(.*?)(?=def [a-zA-Z_]+\()'
    match = re.search(search_function_pattern, content, re.DOTALL)
    
    if match:
        original_function = match.group(0)
        function_body = match.group(1)
        
        # Prepare the fixed function with proper triple quotes for docstring
        fixed_function = '''def search_math_knowledge(query, limit=5):
    """
    Search the Qdrant vector database for relevant mathematical knowledge.
    
    Args:
        query (str): The search query
        limit (int): Maximum number of results to return
        
    Returns:
        list: Relevant knowledge entries
    """
    try:
        # Make sure query is a string
        if not isinstance(query, str):
            if isinstance(query, list):
                # If it's a list, join the elements with spaces
                query = " ".join([str(item) for item in query])
            else:
                # For other types, convert to string
                query = str(query)
                
        query_embedding = get_embedding(query)
        search_result = qdrant_client.search(
            collection_name="math_knowledge",
            query_vector=query_embedding,
            limit=limit
        )
        
        results = []
        for match in search_result:
            payload = match.payload
            if "content" in payload:
                results.append(payload["content"])
        
        return results
    except Exception as e:
        logger.error(f"Error searching math knowledge: {e}")
        return []
'''
        
        # Replace the original function with the fixed one
        modified_content = content.replace(original_function, fixed_function)
        
        # Write back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print("Successfully fixed search_math_knowledge function")
    else:
        print("Could not find search_math_knowledge function")

if __name__ == "__main__":
    fix_search_function("math_agent.py") 