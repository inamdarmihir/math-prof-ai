Here's a technically detailed and exhaustive README for your GitHub project:

---

# Math Prof AI

🚀 **An advanced AI-powered solution** for solving mathematical problems with step-by-step reasoning, supporting a wide range of mathematical domains from basic algebra to advanced calculus.

---

📄 **[Documentation](#)** | 🖥️ **[Streamlit Version](#)** | 🛠️ **[Installation Guide](#getting-started)** | 📊 **[Benchmarking](#benchmarking-system)**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                             Math Prof AI                                 │
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

## Features

✅ **Equation Solving**: Solve algebraic, differential, and other types of equations with step-by-step solutions.
✅ **Calculus**: Calculate derivatives, integrals, limits, and series expansions with detailed explanations.
✅ **Linear Algebra**: Solve matrix operations, systems of equations, eigenvalues, and vector spaces.
✅ **Trigonometry**: Work with trigonometric functions, identities, and equations.
✅ **Vector Analysis**: Perform vector operations, gradient, divergence, curl, and vector field analysis.
✅ **Statistics & Probability**: Solve probability problems, statistical analyses, and distributions.
✅ **Number Theory**: Work with prime numbers, modular arithmetic, and number properties.
✅ **LaTeX Formatting**: Beautiful math rendering with proper LaTeX formatting.
✅ **JEE Benchmarking**: Test and validate agent performance on JEE-level problems.

## 🚀 Getting Started

### 📌 Prerequisites
- Python 3.8 or higher
- OpenAI API key
- Qdrant instance (optional, for knowledge retrieval)

### 📥 Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/inamdarmihir/math-prof-ai.git
    cd math-prof-ai
    ```

2. Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Unix/Linux
    venv\Scripts\activate  # On Windows
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Configure environment variables in `.env` file (use `.env.example` as a template):
    ```bash
    OPENAI_API_KEY=your_openai_api_key
    OPENAI_MODEL=gpt-3.5-turbo
    OPENAI_TEMPERATURE=0.0
    OPENAI_MAX_TOKENS=4000
    QDRANT_URL=your_qdrant_url  # optional
    QDRANT_API_KEY=your_qdrant_api_key  # optional
    QDRANT_COLLECTION=math_knowledge  # optional
    ```

### ▶️ Running the Math Prof AI

Start the Streamlit application:

```bash
streamlit run math_agent.py
```

The application will be available at [http://localhost:8501](http://localhost:8501).

## 🛠️ Core Components

### 🎨 Streamlit Frontend
- User-friendly interface for inputting mathematical questions.
- LaTeX rendering for mathematical expressions.
- Chat-based interface with conversation history.
- Markdown and formatting support.

### 🧠 Processing Engine
- **Query Processing**: Parses and processes user queries.
- **Math Processing**: Uses SymPy for direct symbolic computation.
- **LaTeX Formatting**: Processes and formats LaTeX expressions for proper display.
- **PII Protection**: Built-in guardrails to protect personally identifiable information.

### 🌍 External Services
- **OpenAI Integration**: Leverages GPT models for complex problem-solving.
- **Knowledge Retrieval**: Optional Qdrant vector database for relevant mathematical knowledge.

### 📊 Benchmarking System
- Comprehensive evaluation framework for testing mathematical capabilities.
- Performance metrics tracking and reporting.

## 🔧 Troubleshooting

### ❌ Common Issues

1. **OpenAI API Key Issues**
   - Ensure your API key is correctly set in the `.env` file.
   - Check for API usage limits or billing issues.

2. **LaTeX Rendering Problems**
   - The application has built-in fixes for common LaTeX issues.
   - For persistent problems, try reformatting your equation.

3. **Performance Considerations**
   - For complex problems, the application may take longer to process.
   - Consider using a more powerful OpenAI model for difficult problems.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to improve the Math Prof AI.

### ⚙️ Development Setup

1. Clone the repository.
2. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
3. Run tests:
   ```bash
   pytest
   ```

### 📏 Code Style
This project follows PEP 8 guidelines. Please ensure your contributions adhere to these standards.

## 📜 License
This project is licensed under the terms of the included LICENSE file.

## 🙌 Acknowledgments

- OpenAI for providing the language model capabilities.
- SymPy for symbolic mathematics functionality.
- Streamlit for the web interface framework.
- Qdrant for vector database capabilities.
- JEE examination board for inspiring the benchmark problems.

---

This README provides a comprehensive overview of the Math Prof AI project, including its architecture, features, installation instructions, and guidelines for contributing. It is designed to be informative and engaging, helping users understand and navigate the project effectively.
