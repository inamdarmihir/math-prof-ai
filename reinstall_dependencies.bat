@echo off
echo Activating virtual environment...
call .venv\Scripts\activate

echo Uninstalling existing LangChain packages...
pip uninstall -y langchain langchain-openai langchain-community langchain-core

echo Installing dependencies with pinned versions...
pip install -r requirements.txt

echo Installation complete!
echo Run 'streamlit run math_agent.py' to start the application. 