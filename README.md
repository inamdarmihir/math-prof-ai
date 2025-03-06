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

### Latest Quick Benchmark Results (March 2025)

<div align="center">

| Metric | Value | Notes |
|--------|-------|-------|
| Overall Accuracy | 80.0% | 4 correct out of 5 problems |
| Problems Tested | 5 | Small test set focused on core concepts |
| Average Confidence | 79.4% | On a scale of 0-100% |
| Average Response Time | 18.22 seconds | Using GPT-4-turbo-preview |

</div>

These results demonstrate significant improvement in accuracy and confidence compared to our previous tests with a smaller, focused test set.

### Expanded Benchmark Results (March 2025)

<div align="center">

| Metric | Value | Notes |
|--------|-------|-------|
| Overall Accuracy | 46.4% | 13 correct out of 28 problems |
| Problems Tested | 28 | Comprehensive evaluation across 9 mathematical domains |
| Average Confidence | 59.5% | On a scale of 0-100% |
| Average Response Time | 20.74 seconds | Using GPT-3.5-turbo |

</div>

### Topic-wise Performance

<div align="center">

| Topic | Accuracy | Confidence | Notes |
|-------|----------|------------|-------|
| Algebra | 80.0% | 78.0% | Best performing domain (4/5 correct) |
| Trigonometry | 75.0% | 72.7% | Strong performance (3/4 correct) |
| Calculus | 50.0% | 58.1% | Mixed results (3/6 correct) |
| Differential Equations | 50.0% | 76.0% | High confidence (1/2 correct) |
| Complex Numbers | 50.0% | 80.0% | Very high confidence (1/2 correct) |
| 3D Geometry | 50.0% | 54.3% | Medium confidence (1/2 correct) |
| Coordinate Geometry | 0.0% | 34.8% | Struggled significantly (0/3 correct) |
| Vectors | 0.0% | 36.0% | Struggled significantly (0/2 correct) |
| Probability | 0.0% | 20.0% | Lowest confidence and performance (0/2 correct) |

</div>

### Difficulty-wise Performance

<div align="center">

| Difficulty | Accuracy | Problems |
|------------|----------|----------|
| Easy | 50.0% | 2/4 correct |
| Medium | 52.9% | 9/17 correct |
| Hard | 28.6% | 2/7 correct |

</div>

### Analysis and Insights

The benchmark results reveal several insights about the Math Agent's capabilities:

1. **Strongest in Algebra and Trigonometry**: The agent performs exceptionally well in algebraic problems and trigonometric calculations, with high accuracy and confidence.

2. **Challenges with Geometry and Probability**: The agent struggles most with coordinate geometry, vectors, and probability problems, suggesting these areas need improvement.

3. **Confidence Correlation**: In most cases, the agent's confidence correlates with its accuracy, with notable exceptions in Differential Equations and Complex Numbers where confidence exceeds performance.

4. **Difficulty Scaling**: Performance declines as problem difficulty increases, with a significant drop for hard problems (28.6% accuracy).

We are working on enhancing the Math Agent's capabilities in the weaker domains and improving its performance on more challenging problems.

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