#!/usr/bin/env python
"""
Run focused benchmarks on identified weak areas: coordinate geometry and hard problems.
"""

import os
import argparse
import subprocess
import logging
import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run focused benchmarks on weak areas')
    parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                        help='Directory to store benchmark results')
    parser.add_argument('--model', type=str, default='gpt-4-turbo-preview',
                        help='Model to use for benchmarking')
    parser.add_argument('--max-problems', type=int, default=10,
                        help='Maximum number of problems to test')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Current timestamp for filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run coordinate geometry benchmark
    logger.info("Running coordinate geometry benchmark...")
    coord_geom_cmd = [
        "python", "jee_enhanced_benchmark.py",
        f"--output={args.output_dir}/coord_geom_{timestamp}.json",
        f"--model={args.model}",
        f"--max-problems={args.max_problems}",
        "--topic=coordinate geometry"
    ]
    
    try:
        subprocess.run(coord_geom_cmd, check=True)
        logger.info(f"Coordinate geometry benchmark completed. Results saved to {args.output_dir}/coord_geom_{timestamp}.json")
    except subprocess.CalledProcessError as e:
        logger.error(f"Coordinate geometry benchmark failed with error: {str(e)}")
    
    # Run hard problems benchmark
    logger.info("Running hard problems benchmark...")
    hard_problems_cmd = [
        "python", "jee_enhanced_benchmark.py",
        f"--output={args.output_dir}/hard_problems_{timestamp}.json",
        f"--model={args.model}",
        f"--max-problems={args.max_problems}",
        "--difficulty=Hard"
    ]
    
    try:
        subprocess.run(hard_problems_cmd, check=True)
        logger.info(f"Hard problems benchmark completed. Results saved to {args.output_dir}/hard_problems_{timestamp}.json")
    except subprocess.CalledProcessError as e:
        logger.error(f"Hard problems benchmark failed with error: {str(e)}")
    
    # Run analysis on all results
    logger.info("Running analysis on all benchmark results...")
    analysis_cmd = [
        "python", "analyze_benchmarks.py",
        f"--results-dir={args.output_dir}",
        "--output=focused_benchmark_analysis.md"
    ]
    
    try:
        subprocess.run(analysis_cmd, check=True)
        logger.info("Analysis completed. Results saved to focused_benchmark_analysis.md")
    except subprocess.CalledProcessError as e:
        logger.error(f"Analysis failed with error: {str(e)}")
    
    logger.info("All focused benchmarks completed.")

if __name__ == "__main__":
    main() 