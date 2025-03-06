#!/usr/bin/env python
import os
import sys
import argparse
import subprocess
import time

def main():
    """
    Run both the original and enhanced benchmark for comparison.
    """
    parser = argparse.ArgumentParser(description="Run Math Agent benchmarks")
    parser.add_argument('--max-problems', type=int, default=5, 
                       help='Maximum number of problems to test')
    parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                       help='Directory to save output files')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 50)
    print("RUNNING ORIGINAL BENCHMARK")
    print("=" * 50)
    
    # Run original benchmark with its supported parameters
    orig_cmd = [
        sys.executable, 
        "jee_benchmark.py", 
        f"--max-problems={args.max_problems}",
        f"--output={args.output_dir}/original_results.json",
    ]
    
    subprocess.run(orig_cmd, check=True)
    
    # Wait a few seconds to let files be written
    time.sleep(3)
    
    print("\n" + "=" * 50)
    print("RUNNING ENHANCED BENCHMARK")
    print("=" * 50)
    
    # Run enhanced benchmark with the same problems
    enh_cmd = [
        sys.executable, 
        "jee_enhanced_benchmark.py", 
        f"--max-problems={args.max_problems}",
        f"--output-dir={args.output_dir}",
        "--output-file=enhanced_results.json",
        "--report-file=enhanced_report.html"
    ]
    
    subprocess.run(enh_cmd, check=True)
    
    print("\n" + "=" * 50)
    print(f"BENCHMARK COMPLETE - Results in {args.output_dir}/")
    print("=" * 50)
    
if __name__ == "__main__":
    main() 