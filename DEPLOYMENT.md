# Deployment Guide for Math Agent

This guide explains how to deploy the Math Agent either locally or to a cloud service.

## Local Deployment

### Prerequisites

- Python 3.8 or higher
- Git
- OpenAI API key
- Optional: Qdrant instance for knowledge retrieval

### Setup Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/math-agent.git
   cd math-agent
   ```

2. **Create and activate a virtual environment**

   On Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   On macOS/Linux:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   For development:
   ```bash
   pip install -e .
   ```

4. **Configure environment variables**

   Copy the example environment file and edit it with your credentials:
   ```bash
   cp .env.example .env
   ```
   
   Then edit the `.env` file to add your OpenAI API key and other settings.

5. **Run the application**

   ```bash
   streamlit run math_agent.py
   ```

   The app will be accessible at http://localhost:8501

## Cloud Deployment

### Deploying to Streamlit Cloud

1. **Push your code to GitHub**
   
   Create a GitHub repository and push your code, making sure not to include your `.env` file:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/math-agent.git
   git push -u origin main
   ```

2. **Set up on Streamlit Cloud**

   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Connect your GitHub repository
   - Add your secrets in the Streamlit Cloud dashboard:
     - Go to Advanced Settings > Secrets
     - Add each variable from your `.env` file

3. **Deploy**

   - Click "Deploy" in the Streamlit Cloud dashboard
   - Your app will be available at a URL provided by Streamlit

### Deploying to Heroku

1. **Prepare your app for Heroku**

   Create a `Procfile` in the root directory (already included in this repo):
   ```
   web: streamlit run math_agent.py
   ```

2. **Create a Heroku app**

   ```bash
   heroku create math-agent-app
   ```

3. **Set environment variables**

   ```bash
   heroku config:set OPENAI_API_KEY=your_openai_api_key
   heroku config:set OPENAI_MODEL=gpt-3.5-turbo
   # Set other variables as needed
   ```

4. **Push to Heroku**

   ```bash
   git push heroku main
   ```

5. **Open your app**

   ```bash
   heroku open
   ```

## Docker Deployment

A Dockerfile is provided for containerized deployment.

1. **Build the Docker image**

   ```bash
   docker build -t math-agent .
   ```

2. **Run the container**

   ```bash
   docker run -p 8501:8501 --env-file .env math-agent
   ```

3. **Access the application**

   The app will be accessible at http://localhost:8501

## JEE Benchmarking

To run the JEE benchmarking tool:

```bash
python jee_benchmark.py
```

This will generate a detailed HTML report of the Math Agent's performance on JEE-level problems.

## Troubleshooting

- **API Key Issues**: Ensure your OpenAI API key is valid and has sufficient quota
- **Dependency Problems**: Try running `pip install -r requirements.txt --upgrade` to update all dependencies
- **Memory Issues**: If the application crashes due to memory constraints, try reducing the model context size in the `.env` file by setting `MODEL_MAX_TOKENS` to a lower value 