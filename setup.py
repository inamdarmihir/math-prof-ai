#!/usr/bin/env python3
"""
Setup script for Math Agent
"""

from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    # Remove comments and empty lines
    requirements = [line for line in requirements 
                   if line and not line.startswith('#') and not line.startswith(' ')]

# Read long description from README
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="math-agent",
    version="1.0.0",
    description="AI-powered Math Assistant for solving mathematical problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Math Agent Team",
    author_email="example@example.com",
    url="https://github.com/yourusername/math-agent",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "math-agent=math_agent:main",
            "jee-benchmark=jee_benchmark:main",
        ],
    },
    include_package_data=True,
) 