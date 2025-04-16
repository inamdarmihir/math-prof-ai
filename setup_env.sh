#!/bin/bash

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Get Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install basic packages
echo "Upgrading pip and installing basic packages..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if all dependencies are installed correctly
echo "Verifying installation..."
python3 -c "import streamlit, sympy, dotenv, openai, qdrant_client, numpy, pandas, matplotlib, scipy, tqdm, requests, tiktoken, aiohttp, langgraph, langsmith, plotly" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "All dependencies installed successfully!"
else
    echo "Some dependencies might be missing. Please check the error messages above."
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOL
# LangSmith Configuration
LANGCHAIN_API_KEY=your_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=math-agent-langgraph

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant Configuration
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
EOL
    echo "Created .env file. Please update it with your API keys."
fi

echo "Setup complete! To activate the virtual environment, run:"
echo "source venv/bin/activate" 