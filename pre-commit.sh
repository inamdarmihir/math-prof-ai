#!/bin/bash
# Pre-commit hook to check for sensitive data before commits
# Copy this file to .git/hooks/pre-commit and make it executable with:
# chmod +x .git/hooks/pre-commit

echo "Running pre-commit checks..."

# Check for API keys in staged files
echo "Checking for API keys in staged files..."
if git diff --cached --name-only | xargs grep -l "sk-[a-zA-Z0-9]\{20,\}" > /dev/null; then
  echo "ERROR: Potential OpenAI API key found in staged files."
  echo "Please remove API keys before committing."
  exit 1
fi

# Check for .env files
echo "Checking for .env files..."
if git diff --cached --name-only | grep -E '\.env$' > /dev/null; then
  echo "ERROR: .env file is staged for commit."
  echo "Please remove .env files from staging before committing."
  exit 1
fi

# Check for backup files
echo "Checking for backup files..."
if git diff --cached --name-only | grep -E '\.(bak|backup|old|tmp|log)$' > /dev/null; then
  echo "WARNING: Backup or log files are staged for commit."
  echo "Do you want to continue? (y/n)"
  read answer
  if [ "$answer" != "y" ]; then
    exit 1
  fi
fi

echo "Pre-commit checks passed!"
exit 0 