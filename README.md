# 🧮 Math Agent

An AI-powered assistant for solving complex math problems with step-by-step solutions, powered by LangGraph, Exa Search, and Streamlit.

## Features

- **Step-by-Step Solutions**: Get detailed explanations of how each math problem is solved
- **Comprehensive Math Support**:
  - Equation solving
  - Derivatives and integrals
  - Systems of equations
  - Limits and series
  - Matrix operations
- **Enhanced Learning**:
  - Related math concepts from trusted sources
  - Visual explanations
  - Performance metrics and observability
- **Advanced Features**:
  - LaTeX support for mathematical notation
  - Exa search integration for related concepts
  - LangSmith tracing for debugging
  - Performance benchmarking

### Improved System of Equations Handling

The latest version includes enhanced capabilities for solving systems of linear equations:

- Automatically detects multiple equations or equations in LaTeX cases environment
- Robust parsing of variables and coefficients across different formats
- Multiple solution methods including SymPy symbolic solving and Cramer's rule
- Elegant display of solutions with LaTeX rendering
- Detailed step-by-step explanation of the solution process
- Handles systems of any size, with special optimization for 2×2 systems

Example: Solve the system `4x + y = 0` and `3x - y = 2`

## Installation

1. Clone the repository:
   ```bash
git clone https://github.com/yourusername/math-prof-ai.git
cd math-prof-ai
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
```env
EXA_API_KEY=your_exa_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run math_agent_langgraph.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Enter a math problem using LaTeX notation, for example:
   - Solve the equation: `$x^2 + 2x + 1 = 0$`
   - Find the derivative: `$\frac{d}{dx}(x^2 + 3x + 2)$`
   - Solve the system: `$x + y = 5$ and $2x - y = 1$`

## Deployment

### Local Deployment

1. Install the required packages:
```bash
pip install streamlit watchdog
```

2. Run the application:
```bash
streamlit run math_agent_langgraph.py
```

### Cloud Deployment

#### Streamlit Cloud

1. Create a Streamlit account at https://streamlit.io/cloud

2. Connect your GitHub repository

3. Configure the deployment:
   - Set the main file path to `math_agent_langgraph.py`
   - Add your environment variables in the Secrets section

4. Deploy the application

#### Docker Deployment

1. Build the Docker image:
```bash
docker build -t math-agent .
```

2. Run the container:
```bash
docker run -p 8501:8501 math-agent
```

## Development

### Project Structure

```
math-prof-ai/
├── math_agent_langgraph.py    # Main application file
├── requirements.txt           # Project dependencies
├── .env                       # Environment variables
└── README.md                  # Project documentation
```

### Testing

Run the test suite:
   ```bash
python -m unittest test_math_agent.py -v
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) for the workflow engine
- [Exa](https://exa.ai) for semantic search capabilities
- [Streamlit](https://streamlit.io) for the web interface
- [SymPy](https://www.sympy.org) for symbolic mathematics