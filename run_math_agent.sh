#!/bin/bash
# Startup script for the Math Agent on Unix/Mac

# Check if virtual environment exists
if [ -d "venv" ]; then
  echo "Activating virtual environment..."
  source venv/bin/activate
else
  echo "Virtual environment not found. Creating one..."
  python3 -m venv venv
  source venv/bin/activate
  echo "Installing dependencies..."
  pip install -r requirements.txt
fi

# Check for .env file
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  echo "Creating .env file from example..."
  cp .env.example .env
  echo "Please edit .env to add your API keys"
fi

# Start the Streamlit app
echo "Starting Math Agent..."
streamlit run math_agent.py 