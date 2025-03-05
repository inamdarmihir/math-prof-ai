#!/usr/bin/env python3
"""
Initialization script for the Math Agent
This script helps set up the Math Agent environment for first-time users.
"""

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_step(step, text):
    """Print a step with its number."""
    print(f"\n[{step}] {text}")

def run_command(command):
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                               capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"Error output: {e.stderr}")
        return None

def create_environment():
    """Create and activate a virtual environment."""
    print_step(1, "Creating a virtual environment")
    
    # Check if virtualenv is installed
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "virtualenv"], 
                      check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing virtualenv: {e}")
        return False
    
    # Create the virtual environment
    env_name = "venv"
    if not os.path.exists(env_name):
        try:
            subprocess.run([sys.executable, "-m", "virtualenv", env_name], 
                         check=True, capture_output=True)
            print(f"  ✓ Virtual environment '{env_name}' created successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e}")
            return False
    else:
        print(f"  ✓ Virtual environment '{env_name}' already exists")
    
    # Provide activation instructions
    if platform.system() == "Windows":
        activate_cmd = f"{env_name}\\Scripts\\activate"
    else:
        activate_cmd = f"source {env_name}/bin/activate"
    
    print(f"\n  To activate the environment, run:")
    print(f"  {activate_cmd}")
    
    return True

def install_dependencies():
    """Install the required dependencies."""
    print_step(2, "Installing dependencies")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                     check=True, capture_output=True)
        print(f"  ✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def setup_environment_vars():
    """Set up environment variables."""
    print_step(3, "Setting up environment variables")
    
    env_example = ".env.example"
    env_file = ".env"
    
    if not os.path.exists(env_example):
        print(f"  ✗ Environment example file '{env_example}' not found")
        return False
    
    if os.path.exists(env_file):
        print(f"  ✓ Environment file '{env_file}' already exists")
        overwrite = input("  Do you want to overwrite it? (y/n): ").lower() == 'y'
        if not overwrite:
            print("  Keeping existing environment file")
            return True
    
    # Copy the example file
    shutil.copy(env_example, env_file)
    print(f"  ✓ Environment file '{env_file}' created from example")
    
    # Prompt for OpenAI API key
    openai_key = input("\n  Enter your OpenAI API key (press Enter to skip): ").strip()
    if openai_key:
        # Read the .env file
        with open(env_file, 'r') as f:
            env_content = f.read()
        
        # Replace the placeholder with the actual key
        env_content = env_content.replace("OPENAI_API_KEY=your_openai_api_key", 
                                         f"OPENAI_API_KEY={openai_key}")
        
        # Write the updated content back
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print("  ✓ OpenAI API key added to environment file")
    
    print(f"\n  Please edit the '{env_file}' file to add any other required settings.")
    return True

def setup_git_hooks():
    """Set up Git hooks for pre-commit checks."""
    print_step(4, "Setting up Git hooks")
    
    git_dir = ".git"
    hooks_dir = os.path.join(git_dir, "hooks")
    
    if not os.path.exists(git_dir):
        print("  ✗ Not a Git repository. Skipping hook setup.")
        return False
    
    if not os.path.exists(hooks_dir):
        os.makedirs(hooks_dir)
    
    # Copy the pre-commit hook
    if platform.system() == "Windows":
        hook_source = "pre-commit.bat"
        hook_dest = os.path.join(hooks_dir, "pre-commit.bat")
    else:
        hook_source = "pre-commit.sh"
        hook_dest = os.path.join(hooks_dir, "pre-commit")
    
    if os.path.exists(hook_source):
        shutil.copy(hook_source, hook_dest)
        
        # Make the hook executable on Unix-like systems
        if platform.system() != "Windows":
            os.chmod(hook_dest, 0o755)
        
        print(f"  ✓ Pre-commit hook installed to {hook_dest}")
    else:
        print(f"  ✗ Pre-commit hook source '{hook_source}' not found")
    
    return True

def main():
    """Main function to initialize the Math Agent."""
    print_header("Math Agent Initialization")
    
    print("This script will help you set up the Math Agent environment.")
    print("It will perform the following steps:")
    print("  1. Create a virtual environment")
    print("  2. Install dependencies")
    print("  3. Set up environment variables")
    print("  4. Set up Git hooks")
    
    proceed = input("\nDo you want to proceed? (y/n): ").lower() == 'y'
    if not proceed:
        print("Initialization cancelled.")
        return
    
    # Run the initialization steps
    create_environment()
    install_dependencies()
    setup_environment_vars()
    setup_git_hooks()
    
    print_header("Initialization Complete")
    print("\nTo start using the Math Agent:")
    
    if platform.system() == "Windows":
        print("1. Activate the environment: venv\\Scripts\\activate")
        print("2. Run the application: streamlit run math_agent.py")
    else:
        print("1. Activate the environment: source venv/bin/activate")
        print("2. Run the application: streamlit run math_agent.py")
    
    print("\nOr use the provided convenience script:")
    print("run_math_agent.bat (Windows) or ./run_math_agent.sh (Unix/Mac)")
    
    print("\nFor JEE benchmarking, run:")
    print("python jee_benchmark.py")
    
    print("\nThank you for using Math Agent!")

if __name__ == "__main__":
    main() 