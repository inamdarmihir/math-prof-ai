#!/usr/bin/env python3
"""
Enhanced JEE Benchmarking Tool for Math Agent

This script tests the improved math agent against JEE (Joint Entrance Examination) level problems,
with special focus on previously weak areas (geometry, vectors, probability) and 
provides comprehensive performance metrics.
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import re

# Import the process_query function from enhanced math agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from math_agent_enhanced import process_query, determine_math_domain

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load problems from the original JEE benchmark file
def load_original_problems():
    try:
        # First try to import from jee_benchmark.py
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from jee_benchmark import JEE_PROBLEMS
        return JEE_PROBLEMS
    except ImportError:
        logger.warning("Could not import JEE_PROBLEMS from jee_benchmark.py")
        # Fallback to loading from a previous results file
        try:
            with open('jee_expanded_benchmark_results.json', 'r') as f:
                data = json.load(f)
                return data.get('problems', [])
        except (FileNotFoundError, json.JSONDecodeError):
            logger.error("Could not load problems from jee_expanded_benchmark_results.json")
            return []

# Additional problems focusing on weak areas (geometry, vectors, probability)
ADDITIONAL_PROBLEMS = [
    # Additional Coordinate Geometry Problems
    {
        "id": "GEOM-4",
        "question": "Find the area of the triangle whose vertices are at (1, 1), (4, 5), and (7, 2).",
        "expected_answer": "15",
        "topic": "Coordinate Geometry",
        "subtopic": "Area of Triangle",
        "difficulty": "Medium"
    },
    {
        "id": "GEOM-5",
        "question": "The point P divides the line joining A(3, 4) and B(9, 7) in the ratio 2:1. Find the coordinates of P.",
        "expected_answer": "(5, 5)",
        "topic": "Coordinate Geometry",
        "subtopic": "Section Formula",
        "difficulty": "Easy"
    },
    {
        "id": "GEOM-6",
        "question": "A circle has its center at (3, 4) and passes through the point (6, 8). Find the equation of the circle.",
        "expected_answer": "(x-3)^2 + (y-4)^2 = 25",
        "topic": "Coordinate Geometry",
        "subtopic": "Circle Equation",
        "difficulty": "Medium"
    },
    
    # Additional Vector Problems
    {
        "id": "VEC-3",
        "question": "If a = i + 2j - k and b = 2i - j + 3k, find a × b.",
        "expected_answer": "5i + 5j + 5k",
        "topic": "Vectors",
        "subtopic": "Cross Product",
        "difficulty": "Medium"
    },
    {
        "id": "VEC-4",
        "question": "Find the scalar triple product of a = i + j + k, b = 2i - j + 3k, and c = 3i + 4j - 2k.",
        "expected_answer": "-25",
        "topic": "Vectors",
        "subtopic": "Scalar Triple Product",
        "difficulty": "Hard"
    },
    
    # Additional Probability Problems
    {
        "id": "PROB-3",
        "question": "A fair die is rolled twice. What is the probability that the sum of the two rolls is 8?",
        "expected_answer": "5/36",
        "topic": "Probability",
        "subtopic": "Dice Probability",
        "difficulty": "Easy"
    },
    {
        "id": "PROB-4",
        "question": "In a box, there are 10 red balls and 15 blue balls. Three balls are drawn at random without replacement. What is the probability that all three balls are of the same color?",
        "expected_answer": "13/92",
        "topic": "Probability",
        "subtopic": "Without Replacement",
        "difficulty": "Medium"
    },
    {
        "id": "PROB-5",
        "question": "The probability that a student passes a mathematics exam is 0.6, and the probability that they pass a physics exam is 0.8. If the probability that they pass at least one exam is 0.9, what is the probability that they pass both exams?",
        "expected_answer": "0.5",
        "topic": "Probability",
        "subtopic": "Conditional Probability",
        "difficulty": "Hard"
    }
]

def normalize_answer(answer):
    """
    Normalize an answer to facilitate comparison.
    
    Args:
        answer: The answer to normalize
        
    Returns:
        Normalized answer string
    """
    if not answer:
        return ""
        
    # Convert answer to string if needed
    answer_str = str(answer)
    
    # Remove whitespace, convert to lowercase
    normalized = re.sub(r'\s+', '', answer_str.lower())
    
    # Replace common equivalents
    replacements = {
        # Common fractions
        '1/2': '0.5',
        '1/3': '0.33333',
        '2/3': '0.66667',
        '1/4': '0.25',
        '3/4': '0.75',
        
        # Math symbols
        '^': '**',
        '×': '*',
        '·': '*',
        
        # LaTeX patterns
        r'\frac{': '(',
        r'}{': ')/',
        r'}': ')',
        r'\sqrt{': 'sqrt(',
        r'\sqrt': 'sqrt',
        r'\pi': '3.14159'
    }
    
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    
    # Special case for square roots
    normalized = re.sub(r'sqrt\((\d+)\)', lambda m: str(float(m.group(1))**0.5), normalized)
    
    # Attempt to evaluate simple expressions
    try:
        # Use eval only for safe numerical expressions
        if re.match(r'^[\d\.\+\-\*\/\(\)\s]+$', normalized):
            evaluated = str(eval(normalized))
            return evaluated
    except:
        pass
        
    return normalized

def check_answer(actual, expected):
    """
    Check if the actual answer matches the expected answer.
    
    Args:
        actual: The actual answer from the Math Agent
        expected: The expected answer
        
    Returns:
        Tuple of (is_correct, keyword_match_ratio)
    """
    try:
        # Normalize answers for comparison
        actual_norm = normalize_answer(actual)
        expected_norm = normalize_answer(expected)
        
        # Direct match check
        if actual_norm == expected_norm:
            return True, 1.0
            
        # Check for the answer in a Final Answer or Result section
        answer_pattern = r'(?:final answer|result|answer)[:\s]+([^.\n]+)'
        answer_matches = re.findall(answer_pattern, actual_norm, re.IGNORECASE)
        
        for answer_match in answer_matches:
            if normalize_answer(answer_match) == expected_norm:
                return True, 1.0
                
        # Extract numerical values for comparison
        actual_nums = re.findall(r'-?\d+\.?\d*', actual_norm)
        expected_nums = re.findall(r'-?\d+\.?\d*', expected_norm)
        
        # Check if numerical values match
        if actual_nums and expected_nums:
            # Convert strings to floats for comparison
            actual_floats = [float(num) for num in actual_nums]
            expected_floats = [float(num) for num in expected_nums]
            
            # Check if any actual number matches any expected number
            for actual_float in actual_floats:
                for expected_float in expected_floats:
                    # Allow for small floating-point differences
                    if abs(actual_float - expected_float) < 0.001:
                        return True, 0.9
        
        # Keyword matching for more complex answers
        expected_keywords = set(re.findall(r'\b\w+\b', expected_norm.lower()))
        actual_keywords = set(re.findall(r'\b\w+\b', actual_norm.lower()))
        
        if expected_keywords and actual_keywords:
            common_keywords = expected_keywords.intersection(actual_keywords)
            keyword_match_ratio = len(common_keywords) / len(expected_keywords)
            
            # Consider high keyword match as correct
            if keyword_match_ratio > 0.8:
                return True, keyword_match_ratio
                
        return False, 0.0
        
    except Exception as e:
        logger.error(f"Error checking answer: {str(e)}")
        return False, 0.0

def benchmark_problems(problems, args):
    """
    Run the benchmark on the specified problems and return the results.
    
    Args:
        problems: List of problem dictionaries
        args: Command-line arguments
    
    Returns:
        Dictionary with benchmark results
    """
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "problems_tested": len(problems),
        "model_used": os.environ.get("OPENAI_MODEL", "gpt-4-turbo-preview"),
        "correct_count": 0,
        "total_confidence": 0.0,
        "total_time": 0.0,
        "problem_results": []
    }
    
    # Import the math_agent to test the actual system, not just direct API calls
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from math_agent import process_query as original_process_query
    
    # Create progress bar
    pbar = tqdm(problems, desc="Testing JEE problems")
    
    for problem in pbar:
        problem_id = problem.get("id", "Unknown")
        question = problem.get("question", "")
        expected_answer = problem.get("answer", "")
        topic = problem.get("topic", "Unknown")
        subtopic = problem.get("subtopic", topic)
        difficulty = problem.get("difficulty", "Unknown")
        
        # Use the original math_agent for testing
        start_time = time.time()
        try:
            # Call the original math agent to test the actual system
            response = original_process_query(question)
            processing_time = time.time() - start_time
            
            # Extract answer and confidence
            if isinstance(response, dict):
                # Handle when process_query returns a dict
                answer = response.get("answer", "")
                confidence = response.get("confidence", 0.0)
            else:
                # Handle when process_query returns a string directly
                answer = response
                # Try to extract confidence from answer
                confidence = 0.0
                confidence_pattern = r'Confidence:?\s*(\d+(?:\.\d+)?)%?'
                confidence_match = re.search(confidence_pattern, answer)
                if confidence_match:
                    confidence_str = confidence_match.group(1)
                    try:
                        confidence = float(confidence_str)
                        # Convert percentage to decimal if needed
                        if confidence > 1.0:
                            confidence /= 100.0
                    except ValueError:
                        pass
            
            # Check if the answer is correct
            is_correct, keyword_match_ratio = check_answer(answer, expected_answer)
            
            # Add to results
            results["problem_results"].append({
                "id": problem_id,
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": answer,
                "is_correct": is_correct,
                "confidence": confidence,
                "processing_time": processing_time,
                "topic": topic,
                "subtopic": subtopic,
                "difficulty": difficulty,
                "keyword_match": keyword_match_ratio
            })
            
            if is_correct:
                results["correct_count"] += 1
            
            results["total_confidence"] += confidence
            results["total_time"] += processing_time
            
            # Log the result
            result_msg = f"Processed problem {problem_id}: {'Correct' if is_correct else 'Incorrect'} (Confidence: {confidence:.2f})"
            logger.info(result_msg)
            
        except Exception as e:
            logger.error(f"Error processing problem {problem_id}: {str(e)}")
            results["problem_results"].append({
                "id": problem_id,
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": f"Error: {str(e)}",
                "is_correct": False,
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "topic": topic,
                "subtopic": subtopic,
                "difficulty": difficulty,
                "error": str(e)
            })
    
    # Calculate aggregate metrics
    results["accuracy"] = results["correct_count"] / len(problems) if problems else 0
    results["avg_confidence"] = results["total_confidence"] / len(problems) if problems else 0
    results["avg_time"] = results["total_time"] / len(problems) if problems else 0
    
    # Calculate topic-wise metrics
    topic_metrics = {}
    for result in results["problem_results"]:
        topic = result["topic"]
        if topic not in topic_metrics:
            topic_metrics[topic] = {
                "count": 0,
                "correct": 0,
                "total_confidence": 0.0
            }
        
        topic_metrics[topic]["count"] += 1
        if result["is_correct"]:
            topic_metrics[topic]["correct"] += 1
        topic_metrics[topic]["total_confidence"] += result["confidence"]
    
    # Calculate average metrics per topic
    for topic, metrics in topic_metrics.items():
        metrics["accuracy"] = metrics["correct"] / metrics["count"] if metrics["count"] > 0 else 0
        metrics["avg_confidence"] = metrics["total_confidence"] / metrics["count"] if metrics["count"] > 0 else 0
    
    results["topic_metrics"] = topic_metrics
    
    # Calculate difficulty-wise metrics
    difficulty_metrics = {
        "Easy": {"count": 0, "correct": 0},
        "Medium": {"count": 0, "correct": 0},
        "Hard": {"count": 0, "correct": 0}
    }
    
    for result in results["problem_results"]:
        difficulty = result["difficulty"]
        if difficulty in difficulty_metrics:
            difficulty_metrics[difficulty]["count"] += 1
            if result["is_correct"]:
                difficulty_metrics[difficulty]["correct"] += 1
    
    # Calculate accuracy per difficulty level
    for difficulty, metrics in difficulty_metrics.items():
        metrics["accuracy"] = metrics["correct"] / metrics["count"] if metrics["count"] > 0 else 0
    
    results["difficulty_metrics"] = difficulty_metrics
    
    return results

def generate_html_report(results, output_file):
    """
    Generate an HTML report of benchmark results.
    
    Args:
        results: Benchmark results dictionary
        output_file: Path to save the HTML report
    """
    # Create Pandas DataFrame for easy analysis
    df = pd.DataFrame(results["problem_results"])
    
    # Compute metrics by topic
    topic_metrics = df.groupby("topic").agg({
        "is_correct": "mean",
        "confidence": "mean",
        "id": "count"
    }).reset_index()
    topic_metrics.columns = ["Topic", "Accuracy", "Avg Confidence", "Count"]
    
    # Compute metrics by difficulty
    difficulty_metrics = df.groupby("difficulty").agg({
        "is_correct": "mean",
        "confidence": "mean",
        "id": "count"
    }).reset_index()
    difficulty_metrics.columns = ["Difficulty", "Accuracy", "Avg Confidence", "Count"]
    
    # Generate plots
    plot_dir = "benchmark_plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # Plot accuracy by topic
    plt.figure(figsize=(12, 6))
    bars = plt.bar(topic_metrics["Topic"], topic_metrics["Accuracy"] * 100)
    plt.title("Accuracy by Mathematical Topic", fontsize=16)
    plt.xlabel("Topic", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 100)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    topic_plot_path = os.path.join(plot_dir, "topic_accuracy.png")
    plt.savefig(topic_plot_path)
    plt.close()
    
    # Plot accuracy by difficulty
    plt.figure(figsize=(10, 6))
    bars = plt.bar(difficulty_metrics["Difficulty"], difficulty_metrics["Accuracy"] * 100)
    plt.title("Accuracy by Problem Difficulty", fontsize=16)
    plt.xlabel("Difficulty Level", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.ylim(0, 100)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    difficulty_plot_path = os.path.join(plot_dir, "difficulty_accuracy.png")
    plt.savefig(difficulty_plot_path)
    plt.close()
    
    # Plot confidence vs accuracy
    plt.figure(figsize=(10, 6))
    plt.scatter(df["confidence"], df["is_correct"].astype(int), alpha=0.6)
    plt.title("Confidence vs. Accuracy", fontsize=16)
    plt.xlabel("Confidence Score", fontsize=14)
    plt.ylabel("Correct (1) / Incorrect (0)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    confidence_plot_path = os.path.join(plot_dir, "confidence_vs_accuracy.png")
    plt.savefig(confidence_plot_path)
    plt.close()
    
    # HTML Template
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Math Agent JEE Benchmark Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            text-align: center;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        .summary-box {{
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin-bottom: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 20px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: #3498db;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 1rem;
            color: #7f8c8d;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .plot-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .result-item {{
            margin-bottom: 30px;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
        }}
        .correct {{
            border-left: 5px solid #2ecc71;
        }}
        .incorrect {{
            border-left: 5px solid #e74c3c;
        }}
        .result-header {{
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
            margin-bottom: 10px;
        }}
        .result-id {{
            font-weight: bold;
        }}
        .result-confidence {{
            color: #7f8c8d;
        }}
        .result-topic {{
            background-color: #3498db;
            color: white;
            border-radius: 15px;
            padding: 3px 10px;
            font-size: 0.8rem;
        }}
        .result-difficulty.easy {{
            background-color: #2ecc71;
        }}
        .result-difficulty.medium {{
            background-color: #f39c12;
        }}
        .result-difficulty.hard {{
            background-color: #e74c3c;
        }}
        .result-difficulty {{
            color: white;
            border-radius: 15px;
            padding: 3px 10px;
            font-size: 0.8rem;
            margin-left: 10px;
        }}
        .solution-container {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
        }}
        .timestamp {{
            text-align: right;
            color: #7f8c8d;
            font-size: 0.9rem;
            margin-top: 50px;
        }}
    </style>
</head>
<body>
    <h1>Enhanced Math Agent JEE Benchmark Report</h1>
    
    <div class="summary-box">
        <h2>Executive Summary</h2>
        <p>
            The Enhanced Math Agent was evaluated on {results["problems_tested"]} JEE-level mathematics problems.
            The agent solved {results["correct_count"]} problems correctly, achieving an overall accuracy of
            {results["accuracy"]*100:.1f}%.
        </p>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{results["accuracy"]*100:.1f}%</div>
            <div class="metric-desc">{results["correct_count"]} of {results["problems_tested"]} problems</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Average Confidence</div>
            <div class="metric-value">{results["avg_confidence"]*100:.1f}%</div>
            <div class="metric-desc">Self-reported solution confidence</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Avg. Time per Problem</div>
            <div class="metric-value">{results["avg_time"]:.2f}s</div>
            <div class="metric-desc">Processing time per problem</div>
        </div>
    </div>
    
    <h2>Performance by Topic</h2>
    <div class="plot-container">
        <img src="{topic_plot_path}" alt="Accuracy by Topic" style="max-width:100%;">
    </div>
    <table>
        <tr>
            <th>Topic</th>
            <th>Problems Tested</th>
            <th>Accuracy</th>
            <th>Avg. Confidence</th>
        </tr>
        {"".join(f"<tr><td>{row['Topic']}</td><td>{row['Count']}</td><td>{row['Accuracy']*100:.1f}%</td><td>{row['Avg Confidence']*100:.1f}%</td></tr>" for _, row in topic_metrics.iterrows())}
    </table>
    
    <h2>Performance by Difficulty</h2>
    <div class="plot-container">
        <img src="{difficulty_plot_path}" alt="Accuracy by Difficulty" style="max-width:100%;">
    </div>
    <table>
        <tr>
            <th>Difficulty</th>
            <th>Problems Tested</th>
            <th>Accuracy</th>
            <th>Avg. Confidence</th>
        </tr>
        {"".join(f"<tr><td>{row['Difficulty']}</td><td>{row['Count']}</td><td>{row['Accuracy']*100:.1f}%</td><td>{row['Avg Confidence']*100:.1f}%</td></tr>" for _, row in difficulty_metrics.iterrows())}
    </table>
    
    <h2>Confidence vs. Accuracy</h2>
    <div class="plot-container">
        <img src="{confidence_plot_path}" alt="Confidence vs Accuracy" style="max-width:100%;">
    </div>
    
    <h2>Individual Problem Results</h2>
    {"".join(f"""
    <div class="result-item {'correct' if result['is_correct'] else 'incorrect'}">
        <div class="result-header">
            <div>
                <span class="result-id">{result['id']}</span>
                <span class="result-topic">{result['topic']}</span>
                <span class="result-difficulty {result['difficulty'].lower()}">{result['difficulty']}</span>
            </div>
            <div class="result-confidence">Confidence: {result['confidence']*100:.1f}%</div>
        </div>
        <p><strong>Question:</strong> {result['question']}</p>
        <p><strong>Expected Answer:</strong> {result['expected_answer']}</p>
        <p><strong>Result:</strong> {"Correct ✓" if result['is_correct'] else "Incorrect ✗"}</p>
        <div class="solution-container">
            <p><strong>Solution:</strong></p>
            <pre>{result['actual_answer']}</pre>
        </div>
    </div>
    """ for result in results['problem_results'])}
    
    <div class="timestamp">
        Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>
</body>
</html>
"""
    
    # Write HTML report to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"Report generated at {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced JEE Benchmark Tool for Math Agent')
    parser.add_argument('--output-dir', default='.', help='Directory to save output files')
    parser.add_argument('--output-file', default='jee_enhanced_benchmark_results.json', help='Output JSON file name')
    parser.add_argument('--report-file', default='jee_enhanced_benchmark_results.html', help='Output HTML report file name')
    parser.add_argument('--max-problems', type=int, default=None, help='Maximum number of problems to test')
    parser.add_argument('--focus-weak-areas', action='store_true', help='Focus on previously weak areas (geometry, vectors, probability)')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load original problems
    original_problems = load_original_problems()
    
    # Combine with additional problems focusing on weak areas if requested
    if args.focus_weak_areas:
        all_problems = original_problems + ADDITIONAL_PROBLEMS
        logger.info(f"Testing with focus on weak areas: {len(all_problems)} problems total")
    else:
        all_problems = original_problems
        logger.info(f"Testing standard problem set: {len(all_problems)} problems")
    
    # Run benchmarks
    results = benchmark_problems(all_problems, args)
    
    # Save results to JSON file
    output_path = os.path.join(args.output_dir, args.output_file)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")
    
    # Generate HTML report
    report_path = os.path.join(args.output_dir, args.report_file)
    generate_html_report(results, report_path)
    
    # Print summary to console
    print(f"\nJEE Benchmark Summary:")
    print(f"Problems Tested: {results['problems_tested']}")
    print(f"Accuracy: {results['accuracy']*100:.1f}%")
    print(f"Average Confidence: {results['avg_confidence']*100:.1f}%")
    print(f"Average Response Time: {results['avg_time']:.2f} seconds")
    print(f"Detailed report saved to: {report_path}")

if __name__ == "__main__":
    main() 