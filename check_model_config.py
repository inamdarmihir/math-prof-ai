#!/usr/bin/env python3
"""
Check Model Configuration
This script checks and displays the current model configuration for Math Agent
"""

import os
import sys
import logging
from load_config import load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def check_configuration():
    """Check and display the current configuration for the Math Agent."""
    print("\n=== Math Agent Configuration Checker ===\n")
    
    # Load config
    print("Loading configuration from config.toml...")
    config = load_config()
    
    # Check OpenAI model configuration
    print("\n-- OpenAI Configuration --")
    openai_model = os.getenv("OPENAI_MODEL")
    if openai_model:
        print(f"✓ Model configured: {openai_model}")
    else:
        print("✗ No model configured in environment")
        
        # Check config directly
        if config and 'openai' in config and 'model' in config['openai']:
            print(f"  → Found in config.toml: {config['openai']['model']}")
            print("  → Warning: Not loaded into environment variables")
    
    # Check OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        masked_key = openai_api_key[:4] + "..." + openai_api_key[-4:] if len(openai_api_key) > 8 else "***"
        print(f"✓ API key configured: {masked_key}")
    else:
        print("✗ No API key configured in environment")
    
    # Check embedding model
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    print(f"✓ Embedding model: {embedding_model}")
    
    # Check other important configurations
    print("\n-- Other Configurations --")
    
    # LangSmith
    langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
    if langsmith_api_key:
        print("✓ LangSmith API key configured")
        print(f"  → LangSmith project: {os.getenv('LANGCHAIN_PROJECT', 'Not set')}")
        print(f"  → LangSmith tracing: {os.getenv('LANGCHAIN_TRACING_V2', 'Not set')}")
    else:
        print("✗ No LangSmith API key configured")
    
    # Qdrant
    qdrant_url = os.getenv("QDRANT_URL")
    if qdrant_url:
        print(f"✓ Qdrant URL: {qdrant_url}")
    else:
        print("✗ No Qdrant URL configured")
    
    print("\n=== Configuration Check Complete ===\n")

if __name__ == "__main__":
    try:
        check_configuration()
    except Exception as e:
        logging.error(f"Error checking configuration: {str(e)}")
        sys.exit(1) 