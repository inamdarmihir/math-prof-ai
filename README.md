# Math Agent

An advanced AI-powered solution for solving mathematical problems with step-by-step reasoning, supporting a wide range of mathematical domains.

## Features

- **Equation Solving**: Solve algebraic, differential, and other types of equations
- **Calculus**: Calculate derivatives, integrals, and limits
- **Linear Algebra**: Solve matrix operations and systems of equations
- **Trigonometry**: Work with trigonometric functions and identities
- **Vector Analysis**: Perform vector operations and analysis
- **JEE Benchmarking**: Test and validate agent performance on JEE-level problems

## Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Qdrant instance (optional, for knowledge retrieval)

### Installation

1. Clone this repository
2. Create and activate a virtual environment (recommended)
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Configure environment variables in `.env` file (use `.env.example` as a template)
```
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-3.5-turbo  # or other preferred model
QDRANT_URL=your_qdrant_url  # optional
QDRANT_API_KEY=your_qdrant_api_key  # optional
QDRANT_COLLECTION=math_knowledge  # optional
```

### Running the Math Agent

Start the Streamlit application:

```
streamlit run math_agent.py
```

The application will be available at http://localhost:8501 (or another port if 8501 is already in use).

## JEE Benchmarking

The Math Agent includes a comprehensive benchmarking tool for testing its performance on JEE-level math problems.

### Running the Benchmark

For a quick test with 5 problems:

```
run_jee_benchmark.bat  # Windows
```

Or manually:

```
pip install -r requirements_benchmark.txt
python jee_benchmark.py --max-problems 5
```

For the full benchmark:

```
python jee_benchmark.py
```

### Benchmark Features

- Tests across multiple mathematical domains (Calculus, Algebra, Geometry, Trigonometry, etc.)
- Varying difficulty levels (Easy, Medium, Hard)
- Detailed HTML report generation with visualizations
- Performance metrics (accuracy, response time, confidence)

### Sample JEE Problems

The benchmark includes problems similar to those found in the Joint Entrance Examination (JEE), covering topics such as:

- Derivatives and integrals
- Algebraic equations and series
- Coordinate geometry
- Trigonometric identities
- Differential equations
- Vector operations

## Implementation Details

### Core Components

1. **Math Processing**: Utilizes SymPy for direct mathematical computation
2. **LLM Integration**: Leverages OpenAI's GPT models for complex problem-solving
3. **Knowledge Retrieval**: Optional Qdrant vector database for retrieving relevant mathematical knowledge
4. **PII Protection**: Built-in safeguards to protect personally identifiable information

### Architecture

- **Streamlit Frontend**: User-friendly interface for inputting questions and displaying answers
- **Processing Pipeline**: Combines direct computation with LLM-based reasoning
- **Benchmarking System**: Independent evaluation framework for testing mathematical capabilities

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to improve the Math Agent.

## License

This project is licensed under the terms of the included LICENSE file.

## Acknowledgments

- OpenAI for providing the language model capabilities
- SymPy for symbolic mathematics functionality
- Streamlit for the web interface framework
