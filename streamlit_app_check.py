import os
import streamlit as st
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_api_keys():
    """Check if all required API keys are properly accessible"""
    
    results = {}
    
    # Check OpenAI API key
    try:
        openai_api_key = st.secrets["openai"]["api_key"]
        if openai_api_key and len(openai_api_key) > 10:  # Basic validation
            # Try to initialize the client
            client = OpenAI(api_key=openai_api_key)
            results["openai"] = {"status": "SUCCESS", "message": "OpenAI API key is valid and client initialized"}
        else:
            results["openai"] = {"status": "ERROR", "message": "OpenAI API key appears to be empty or too short"}
    except Exception as e:
        results["openai"] = {"status": "ERROR", "message": f"Error accessing OpenAI API key: {str(e)}"}
    
    # Check other API keys
    try:
        # Weaviate
        weaviate_api_key = st.secrets["weaviate"]["api_key"]
        results["weaviate"] = {"status": "SUCCESS", "message": "Weaviate API key accessible"} if weaviate_api_key else {"status": "ERROR", "message": "Weaviate API key not found"}
        
        # Qdrant
        qdrant_api_key = st.secrets["qdrant"]["api_key"]
        results["qdrant"] = {"status": "SUCCESS", "message": "Qdrant API key accessible"} if qdrant_api_key else {"status": "ERROR", "message": "Qdrant API key not found"}
        
        # Exa
        exa_api_key = st.secrets["exa"]["api_key"]
        results["exa"] = {"status": "SUCCESS", "message": "Exa API key accessible"} if exa_api_key else {"status": "ERROR", "message": "Exa API key not found"}
        
        # Serper
        serper_api_key = st.secrets["serper"]["api_key"]
        results["serper"] = {"status": "SUCCESS", "message": "Serper API key accessible"} if serper_api_key else {"status": "ERROR", "message": "Serper API key not found"}
    
    except Exception as e:
        logger.error(f"Error checking API keys: {e}")
        results["other"] = {"status": "ERROR", "message": f"Error checking other API keys: {str(e)}"}
    
    return results

def main():
    st.title("Streamlit Cloud Deployment Check")
    
    st.write("""
    This app checks if your configuration is correctly set up for Streamlit Cloud deployment.
    It verifies that all required API keys are accessible.
    """)
    
    if st.button("Check Configuration"):
        results = check_api_keys()
        
        st.subheader("Configuration Check Results")
        
        for service, result in results.items():
            if result["status"] == "SUCCESS":
                st.success(f"{service}: {result['message']}")
            else:
                st.error(f"{service}: {result['message']}")
        
        st.subheader("Recommended Fix for OpenAI Error")
        st.code("""
# Make sure your code is accessing the API key correctly:
import streamlit as st
from openai import OpenAI

# Get API key from Streamlit secrets
openai_api_key = st.secrets["openai"]["api_key"]

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)
        """, language="python")
        
        st.subheader("Streamlit Secrets Format")
        st.code("""
# This should be in your Streamlit Cloud dashboard under "Secrets"
[openai]
api_key = "your-actual-api-key"
model = "gpt-4-turbo-preview"

[weaviate]
url = "https://r4lc8ocsiucntykapnhma.c0.us-west3.gcp.weaviate.cloud"
grpc_url = "https://grpc-r4lc8ocsiucntykapnhma.c0.us-west3.gcp.weaviate.cloud"
api_key = "your-actual-weaviate-key"

[qdrant]
url = "https://0d6f2682-65bf-42fb-b263-e3b40605ffab.us-west-2-0.aws.cloud.qdrant.io"
api_key = "your-actual-qdrant-key"
collection = "math_knowledge"
        """, language="toml")

if __name__ == "__main__":
    main() 