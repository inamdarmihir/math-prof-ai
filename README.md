# Math Agent

An advanced AI-powered solution for solving mathematical problems with step-by-step reasoning, supporting a wide range of mathematical domains from basic algebra to advanced calculus.

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────┐
│                             Math Agent                                   │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
           ┌─────────────────────────────────────────────┐
           │                                             │
┌──────────▼───────────┐    ┌─────────▼───────────┐    ┌▼──────────────────┐
│   Streamlit Frontend  │    │  Processing Engine  │    │  External Services │
└──────────┬───────────┘    └─────────┬───────────┘    └───────────┬────────┘
           │                          │                            │
┌──────────▼───────────┐    ┌─────────▼───────────┐    ┌───────────▼────────┐
│ - User Input Handling │    │ - Query Processing │    │ - OpenAI API        │
│ - LaTeX Rendering     │    │ - Math Processing  │    │ - Qdrant Vector DB  │
│ - Display Formatting  │    │ - Text Formatting  │    │   (Optional)        │
└──────────┬───────────┘    └─────────┬───────────┘    └───────────┬────────┘
           │                          │                            │
           └─────────────────────────────────────────────┘
                                    │
                         ┌──────────▼─────────┐
                         │  Benchmarking Tool  │
                         └──────────┬─────────┘
                                    │
                         ┌──────────▼─────────┐
                         │ - JEE Problems      │
                         │ - Performance Metrics│
                         │ - Report Generation │
                         └────────────────────┘
```

</div>

## Features

- **Equation Solving**: Solve algebraic, differential, and other types of equations with step-by-step solutions
- **Calculus**: Calculate derivatives, integrals, limits, and series expansions with detailed explanations
- **Linear Algebra**: Solve matrix operations, systems of equations, eigenvalues, and vector spaces
- **Trigonometry**: Work with trigonometric functions, identities, and equations
- **Vector Analysis**: Perform vector operations, gradient, divergence, curl, and vector field analysis
- **Statistics & Probability**: Solve probability problems, statistical analyses, and distributions
- **Number Theory**: Work with prime numbers, modular arithmetic, and number properties
- **LaTeX Formatting**: Beautiful math rendering with proper LaTeX formatting
- **JEE Benchmarking**: Test and validate agent performance on JEE-level problems

## Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Qdrant instance (optional, for knowledge retrieval)

### Installation

1. Clone this repository
    ```bash
    git clone https://github.com/yourusername/math-agent.git
    cd math-agent
    ```

2. Create and activate a virtual environment (recommended)
    ```bash
    python -m venv venv
    # On Unix/Linux:
    source venv/bin/activate  
    # On Windows:
    venv\Scripts\activate
    ```

3. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

4. Configure environment variables in `.env` file (use `.env.example` as a template)
    ```bash
    OPENAI_API_KEY=your_openai_api_key
    OPENAI_MODEL=gpt-3.5-turbo  # or other preferred model
    OPENAI_TEMPERATURE=0.0      # 0.0 for deterministic answers
    OPENAI_MAX_TOKENS=4000      # maximum token limit
    QDRANT_URL=your_qdrant_url  # optional
    QDRANT_API_KEY=your_qdrant_api_key  # optional
    QDRANT_COLLECTION=math_knowledge  # optional
    ```

### Running the Math Agent

Start the Streamlit application:

```bash
streamlit run math_agent.py
```

The application will be available at http://localhost:8501 (or another port if 8501 is already in use).

## Architecture and Implementation Details

### Core Components

The Math Agent is built with a modular architecture consisting of several key components:

1. **Streamlit Frontend**
   - User-friendly interface for inputting mathematical questions
   - LaTeX rendering for mathematical expressions
   - Chat-based interface with conversation history
   - Markdown and formatting support

2. **Processing Engine**
   - **Query Processing**: Parses and processes user queries
   - **Math Processing**: Uses SymPy for direct symbolic computation
   - **LaTeX Formatting**: Processes and formats LaTeX expressions for proper display
   - **PII Protection**: Built-in guardrails to protect personally identifiable information

3. **External Services**
   - **OpenAI Integration**: Leverages GPT models for complex problem-solving
   - **Knowledge Retrieval**: Optional Qdrant vector database for relevant mathematical knowledge

4. **Benchmarking System**
   - Comprehensive evaluation framework for testing mathematical capabilities
   - Performance metrics tracking and reporting

### Key Functions

The Math Agent's implementation includes several specialized functions:

- **LaTeX Processing**:
  - `fix_latex_formatting()`: Applies multiple formatting fixes to LaTeX content
  - `fix_align_content()`: Fixes alignment environments for proper display
  - `fix_nested_delimiters()`: Corrects nested delimiters and fractions
  - `fix_latex_for_streamlit()`: Specifically formats LaTeX for Streamlit display
  - `fix_broken_latex()`: Repairs broken LaTeX formatting in user inputs

- **Mathematical Processing**:
  - `extract_equations()`: Identifies and extracts equations from text
  - `solve_equation()`: Solves algebraic equations using SymPy
  - `calculate_derivative()`: Computes derivatives of mathematical expressions
  - `evaluate_integral()`: Evaluates integrals of mathematical expressions

- **Utility Functions**:
  - `get_embedding()`: Generates vector embeddings for semantic search
  - `search_math_knowledge()`: Retrieves relevant mathematical knowledge
  - `output_guardrails()`: Applies safety checks to model outputs

- **Core Processing**:
  - `process_query()`: Main function processing user queries
  - `format_for_streamlit_display()`: Prepares content for Streamlit display

### Data Flow

1. User inputs a mathematical question through the Streamlit interface
2. The application attempts to directly solve the problem using SymPy
3. If direct solution fails, the query is enhanced with relevant math knowledge
4. The enhanced query is sent to the OpenAI model with specific prompting
5. The response is processed to ensure proper LaTeX formatting
6. The formatted response is displayed to the user with proper math rendering

## JEE Benchmarking

The Math Agent includes a comprehensive benchmarking tool for testing its performance on JEE-level math problems.

### Running the Benchmark

For a quick test with 5 problems:

```bash
# On Windows:
run_jee_benchmark.bat

# On Unix/Linux:
./run_jee_benchmark.sh
```

Or manually:

```bash
pip install -r requirements_benchmark.txt
python jee_benchmark.py --max-problems 5
```

For the full benchmark:

```bash
python jee_benchmark.py
```

### Benchmark Features

- **Comprehensive Test Suite**: Tests across multiple mathematical domains
  - Calculus (Derivatives, Integrals, Limits)
  - Algebra (Equations, Inequalities, Series)
  - Geometry (Coordinate Geometry, Vectors, 3D)
  - Trigonometry (Identities, Equations)
  - Statistics & Probability

- **Difficulty Levels**:
  - Easy: Basic application of formulas
  - Medium: Multi-step problems
  - Hard: Complex problems requiring deep understanding

- **Detailed Reports**:
  - HTML report generation with visualizations
  - Performance metrics (accuracy, response time, confidence)
  - Domain-specific performance analysis
  - Comparison with previous benchmark runs

### Sample JEE Problems

The benchmark includes problems similar to those found in the Joint Entrance Examination (JEE), covering topics such as:

- Derivatives and integrals of complex functions
- Algebraic equations and series expansions
- Coordinate geometry problems
- Trigonometric identities and equations
- Differential equations with specific constraints
- Vector operations and 3D geometry
- Probability and statistics problems

## Advanced Usage

### Custom Math Knowledge Base

You can enhance the Math Agent with your own mathematical knowledge:

1. Create a JSON file with mathematical concepts and examples
2. Use the knowledge ingestion script to add to Qdrant:
    ```bash
    python ingest_knowledge.py --source your_math_knowledge.json
    ```

### API Integration

The Math Agent can be integrated into other applications:

```python
from math_agent import process_query

# Process a mathematical query
result = process_query("Find the derivative of f(x) = x^3 + 3x^2 - 2x + 1")
print(result)
```

### Environment Variables

Additional configuration options:

```
LOG_LEVEL=INFO                   # Logging level (DEBUG, INFO, WARNING, ERROR)
ENABLE_DIRECT_CALCULATION=true   # Enable/disable direct calculation with SymPy
ENABLE_KNOWLEDGE_RETRIEVAL=true  # Enable/disable knowledge retrieval
MAX_HISTORY_MESSAGES=10          # Maximum conversation history to maintain
```

## Troubleshooting

### Common Issues

1. **OpenAI API Key Issues**
   - Ensure your API key is correctly set in the `.env` file
   - Check for API usage limits or billing issues

2. **LaTeX Rendering Problems**
   - The application has built-in fixes for common LaTeX issues
   - For persistent problems, try reformatting your equation

3. **Performance Considerations**
   - For complex problems, the application may take longer to process
   - Consider using a more powerful OpenAI model for difficult problems

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to improve the Math Agent.

### Development Setup

1. Clone the repository
2. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
3. Run tests:
   ```bash
   pytest
   ```

### Code Style

This project follows PEP 8 guidelines. Please ensure your contributions adhere to these standards.

## License

This project is licensed under the terms of the included LICENSE file.

## Acknowledgments

- OpenAI for providing the language model capabilities
- SymPy for symbolic mathematics functionality
- Streamlit for the web interface framework
- Qdrant for vector database capabilities
- JEE examination board for inspiring the benchmark problems

## Roadmap

Future enhancements planned for the Math Agent:

- Additional mathematical domains (Graph Theory, Abstract Algebra)
- Multiple language support
- Interactive problem-solving with step-by-step guidance
- Customizable UI themes
- Mobile-friendly interface
- Integration with educational platforms
- Offline mode with local models
