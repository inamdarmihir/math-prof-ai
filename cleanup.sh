#!/bin/bash
# Cleanup script for math-prof-ai before committing

echo "Cleaning up repository files..."

# Remove Python cache files
echo "Removing __pycache__ directories..."
rm -rf __pycache__
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# Remove temporary files
echo "Removing temporary files..."
find . -name "*.tmp" -delete
find . -name "*.bak" -delete
find . -name "*.swp" -delete

# Remove logs (except the main log file if needed)
echo "Cleaning logs directory..."
mkdir -p logs
rm -f logs/config.log

# Remove test files that are no longer needed
echo "Removing unnecessary test files..."
rm -f test_*.py
rm -f debug_*.py
rm -f *_fix.py
rm -f fix_*.py
rm -f *benchmark*.py
rm -f *_benchmark*.json
rm -f *_benchmark*.html

# Keep essential test files
touch .keep_test_files
git checkout -- test_connections.py
git checkout -- test_api_connections.py

# Remove older versions of the main agent
echo "Removing older agent versions..."
rm -f math_agent.py
rm -f math_agent_enhanced.py
rm -f math_agent.py.*
rm -f math_agent_simplified.py

# Remove JSON test results
echo "Removing test result JSON files..."
rm -f *_test_results.json
rm -f *_results.json

# Remove test markdown files
echo "Removing benchmark analysis files..."
rm -f benchmark_*.md
rm -f *benchmark_analysis.md

# Make the remaining shell scripts executable
echo "Making shell scripts executable..."
chmod +x *.sh

echo "Cleanup complete!"
echo "The following core files are retained:"
echo "- math_agent_langgraph.py (main application)"
echo "- load_config.py (configuration loader)"
echo "- config.toml (configuration file)"
echo "- requirements.txt (dependencies)"
echo "- setup_env.sh (environment setup)"
echo "- README.md (documentation)"
echo "- .env (environment variables - ensure this is in .gitignore)"

echo "Remember to review changes before committing!" 