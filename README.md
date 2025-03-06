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

## Research & Implementation Approach

The Math Agent combines several cutting-edge technologies and research approaches to create a powerful mathematical problem-solving system:

### Research Background

Our work builds on significant research in the following areas:

- **Large Language Models (LLMs)**: We leverage the latest research in prompt engineering and Chain-of-Thought (CoT) reasoning to guide LLMs through complex mathematical problem-solving.
- **Symbolic Mathematics**: Integration with SymPy for precise symbolic computation, applying research in computer algebra systems.
- **Mathematical Reasoning**: Implementation of step-by-step reasoning techniques based on cognitive science research on how humans solve mathematical problems.
- **LaTeX Processing**: Development of specialized LaTeX handling algorithms to improve rendering of complex mathematical expressions.

### Implementation Architecture

The Math Agent's architecture was carefully designed to optimize for both performance and accuracy:

#### Core Processing Engine

1. **Query Understanding Module**:
   - Analyzes user questions to identify mathematical domain and problem type
   - Extracts key variables, equations, and constraints
   - Implements NLP-based intent recognition specific to mathematical language

2. **Mathematical Processing Pipeline**:
   - Custom-built symbolic computation interface with SymPy
   - Step generation algorithm that produces human-readable explanation steps
   - Solution verification through reverse calculation

3. **LaTeX Formatter**:
   - Advanced regex-based LaTeX cleanup and standardization 
   - Special handling for complex structures like matrices, fractions, and systems of equations
   - Alignment environment formatting for multi-step solutions

#### External Knowledge Integration

The system optionally integrates with Qdrant vector database to retrieve:
- Mathematical formulas and theorems
- Step-by-step solution templates
- Domain-specific solution strategies

## JEE Benchmarking Results

We have conducted initial benchmarking of the Math Agent against JEE (Joint Entrance Examination) level problems. While our current test set is small (only 3 problems), it provides valuable insights into the system's capabilities.

### Detailed Benchmark Performance (March 2025)

<div align="center">

| Metric | Value | Notes |
|--------|-------|-------|
| Overall Accuracy | 66.7% | 2 correct out of 3 problems |
| Problems Tested | 3 | Currently expanding test set |
| Average Confidence | 0.70 | On a scale of 0-1 |
| Average Response Time | 21.37 seconds | Using GPT-3.5-turbo |

</div>

### Specific Problem Results

1. **Problem ID: CALC-1**
   - **Domain**: Calculus (Differentiation)
   - **Problem**: "Find the derivative of f(x) = x^3 - 3x^2 + 2x - 1 with respect to x."
   - **Result**: ✓ Correct
   - **Confidence**: 1.0
   - **Time**: 18.92 seconds
   - **Solution Method**: Successfully applied the power rule for each term

2. **Problem ID: CALC-2**
   - **Domain**: Calculus (Integration)
   - **Problem**: "Evaluate the indefinite integral of 2x + sin(x)"
   - **Result**: ✓ Correct
   - **Confidence**: 0.85
   - **Time**: 19.87 seconds
   - **Solution Method**: Applied standard integration formulas

3. **Problem ID: ALG-1**
   - **Domain**: Algebra (Systems of Equations)
   - **Problem**: "Solve the system: 2x + y = 5, 3x - y = 2"
   - **Result**: ✗ Incorrect
   - **Confidence**: 0.24
   - **Time**: 25.32 seconds
   - **Error Analysis**: Arithmetic mistake in elimination step

### Performance Analysis

Our preliminary results show:

- **Strong performance on calculus problems**: The system excels at applying standard differentiation and integration rules.
- **Challenges with algebraic manipulation**: More complex multi-step algebra problems require improved handling.
- **Correlation between confidence and accuracy**: Problems with high confidence scores (>0.8) were consistently solved correctly.

### Benchmark Development Roadmap

1. **Expanding Test Set**: Currently developing a comprehensive set of 100+ JEE-level problems across all mathematical domains.
2. **Comparative Analysis**: Will benchmark against other available math-solving systems.
3. **Error Analysis Framework**: Developing a taxonomy of error types to guide improvement efforts.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- Qdrant instance (optional, for knowledge retrieval)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/inamdarimihr/math-proof-ai.git
   cd math-proof-ai
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate # On Unix/Linux
   venv\Scripts\activate # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables in `.env` file (use `.env.example` as a template):
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_MODEL=gpt-3.5-turbo
   OPENAI_TEMPERATURE=0.
   OPENAI_MAX_TOKENS=400
   QDRANT_URL=your_qdrant_url # optional
   QDRANT_API_KEY=your_qdrant_api_key # optional
   QDRANT_COLLECTION=math_knowledge # optional
   ```

### Running the Math Agent

Start the Streamlit application:

```bash
streamlit run math_agent.py
```

The application will be available at [http://localhost:8501](http://localhost:8501).

## Implementation Details

### Key Components and Modules

#### 1. LaTeX Processing Functions

The Math Agent implements sophisticated LaTeX handling:

```python
def fix_latex_for_streamlit(latex_text):
    """
    Clean and standardize LaTeX for Streamlit rendering with special handling for common patterns.
    """
    # Remove square brackets often added by LLMs
    latex_text = re.sub(r'\[\s*\\begin\{align\}', r'\\begin{align}', latex_text)
    latex_text = re.sub(r'\\end\{align\}\s*\]', r'\\end{align}', latex_text)
    
    # Handle system of equations formatting
    latex_text = re.sub(r'\\begin\{cases\}(.*?)\\end\{cases\}', 
                      lambda m: fix_system_of_equations(m.group(1)), 
                      latex_text, flags=re.DOTALL)
                      
    # Fix fraction formatting
    latex_text = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', 
                      r'\\frac{\1}{\2}', latex_text)
    
    return latex_text
```

#### 2. Mathematical Processing

Our symbolic computation integrates with SymPy for precise results:

```python
def solve_equation(equation_str):
    """
    Parse and solve mathematical equations using SymPy.
    """
    try:
        # Parse the equation
        eq = parse_expr(equation_str)
        # Extract variables
        variables = list(eq.free_symbols)
        # Solve the equation
        solution = solve(eq, variables[0])
        return solution
    except Exception as e:
        return f"Error solving equation: {str(e)}"
```

#### 3. Query Processing Pipeline

The query processing flow implements a multi-stage approach:

1. **Preprocessing**: Clean and normalize the input
2. **Domain Classification**: Identify the mathematical domain 
3. **Query Enhancement**: Add context and structure for the LLM
4. **Response Generation**: Generate detailed step-by-step solutions
5. **Postprocessing**: Format LaTeX and clean up the response

## JEE Benchmarking

The Math Agent includes a comprehensive benchmarking tool for testing its performance on JEE-level math problems. The tool:

1. **Provides a test suite** of carefully crafted JEE-level problems
2. **Evaluates responses** using a combination of:
   - Exact answer matching
   - Mathematical equivalence checking
   - Solution path analysis
3. **Generates detailed reports** with:
   - Overall performance metrics
   - Problem-by-problem analysis
   - Visualizations of performance by domain and difficulty
   - Error analysis and categorization

To run a benchmark test:

```bash
python jee_benchmark.py --output-dir ./benchmark_results
```

The resulting HTML report provides visualization of performance metrics.

## Troubleshooting

### Common Issues

1. **OpenAI API Key Issues**
   - Ensure your API key is correctly set in the `.env` file
   - Check for API usage limits or billing issues

2. **LaTeX Rendering Problems**
   - The application has built-in fixes for common LaTeX issues.
   - For persistent problems, try reformatting your equation

3. **Performance Considerations**
   - For complex problems, the application may take longer to process.
   - Consider using a more powerful OpenAI model for difficult problems.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to improve the Math Prof AI.

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

### License
This project is licensed under the terms of the included LICENSE file.

## Acknowledgments

- OpenAI for providing the language model capabilities
- SymPy for symbolic mathematics functionality.
- Streamlit for the web interface framework.
- Qdrant for vector database capabilities.
- JEE examination board for inspiring the benchmark problems.

---

This README provides a comprehensive overview of the Math Agent AI project, including its architecture, features, installation instructions, and guidelines for contributing. It is designed to be informative and engaging, helping users understand and navigate the project effectively.
