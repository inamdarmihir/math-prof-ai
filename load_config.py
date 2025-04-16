#!/usr/bin/env python3
"""
Configuration loader for Math Agent
This script loads configuration from config.toml and sets environment variables
"""

import os
import logging
import toml
from dotenv import load_dotenv

# Create logs directory if it doesn't exist
logs_dir = 'logs'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'config.log')),
        logging.StreamHandler()
    ]
)

def load_config():
    """
    Load configuration from config.toml file and set environment variables
    
    Returns:
        dict: The loaded configuration
    """
    # First, load environment variables from .env file if it exists
    load_dotenv()
    
    try:
        # Load configuration from config.toml
        config_path = os.path.join(os.path.dirname(__file__), 'config.toml')
        config = toml.load(config_path)
        logging.info(f"Loaded configuration from {config_path}")
        
        # Set OpenAI environment variables if not already set
        if 'openai' in config:
            if 'api_key' in config['openai'] and config['openai']['api_key'] and not os.getenv("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = config['openai']['api_key']
                logging.info("Set OPENAI_API_KEY from config.toml")
            
            # Ensure the model is set in environment variables
            if 'model' in config['openai'] and not os.getenv("OPENAI_MODEL"):
                os.environ["OPENAI_MODEL"] = config['openai']['model']
                logging.info(f"Set OPENAI_MODEL to {config['openai']['model']} from config.toml")
            
            # Set other OpenAI parameters
            if 'temperature' in config['openai'] and not os.getenv("OPENAI_TEMPERATURE"):
                os.environ["OPENAI_TEMPERATURE"] = str(config['openai']['temperature'])
            
            if 'max_tokens' in config['openai'] and not os.getenv("OPENAI_MAX_TOKENS"):
                os.environ["OPENAI_MAX_TOKENS"] = str(config['openai']['max_tokens'])
        
        # Set other API keys if they exist in config and not in environment
        api_services = ['weaviate', 'exa', 'serper', 'qdrant']
        for service in api_services:
            if service in config and 'api_key' in config[service] and config[service]['api_key'] and not os.getenv(f"{service.upper()}_API_KEY"):
                os.environ[f"{service.upper()}_API_KEY"] = config[service]['api_key']
                logging.info(f"Set {service.upper()}_API_KEY from config.toml")
        
        # Set Qdrant URL if needed
        if 'qdrant' in config and 'url' in config['qdrant'] and not os.getenv("QDRANT_URL"):
            os.environ["QDRANT_URL"] = config['qdrant']['url']
        
        return config
    
    except FileNotFoundError:
        logging.warning(f"Config file not found at {os.path.join(os.path.dirname(__file__), 'config.toml')}")
        return {}
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        return {}

if __name__ == "__main__":
    config = load_config()
    print("Configuration loaded and environment variables set.")
    
    # Print current OPENAI_MODEL
    print(f"Current OPENAI_MODEL: {os.getenv('OPENAI_MODEL', 'Not set')}")
    
    # Verify settings
    if os.getenv("OPENAI_API_KEY"):
        print("OpenAI API Key: ✓ (Set)")
    else:
        print("OpenAI API Key: ✗ (Not set)") 