#!/usr/bin/env python3
"""
JEE Benchmarking Tool for Math Agent

This script tests the math agent against JEE (Joint Entrance Examination) level problems
and generates a detailed report of its performance for inclusion in proposal documents.
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

# Import the process_query function from math_agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from math_agent import process_query

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# JEE benchmark problems covering different topics
JEE_PROBLEMS = [
    # Calculus
    {
        "id": "CALC-1",
        "question": "Find the derivative of f(x) = x^3 - 3x^2 + 2x - 1 with respect to x.",
        "expected_answer": "3x^2 - 6x + 2",
        "topic": "Calculus",
        "subtopic": "Differentiation",
        "difficulty": "Easy"
    },
    {
        "id": "CALC-2",
        "question": "Evaluate the indefinite integral: ∫(2x + 3)/(x^2 + 3x + 2) dx",
        "expected_answer": "log(x^2 + 3x + 2) + C",
        "topic": "Calculus",
        "subtopic": "Integration",
        "difficulty": "Medium"
    },
    {
        "id": "CALC-3",
        "question": "Find the area bounded by the curve y = x^2 and the lines y = 1, y = 4, and x = 0.",
        "expected_answer": "7/3",
        "topic": "Calculus",
        "subtopic": "Definite Integration",
        "difficulty": "Hard"
    },
    
    # Algebra
    {
        "id": "ALG-1",
        "question": "Solve the equation: 2x^2 - 5x + 3 = 0",
        "expected_answer": "x = 1 or x = 3/2",
        "topic": "Algebra",
        "subtopic": "Quadratic Equations",
        "difficulty": "Easy"
    },
    {
        "id": "ALG-2",
        "question": "If the sum of the first n terms of an AP is 3n^2 + 5n, then find the nth term of this AP.",
        "expected_answer": "6n + 2",
        "topic": "Algebra",
        "subtopic": "Arithmetic Progression",
        "difficulty": "Medium"
    },
    {
        "id": "ALG-3",
        "question": "Find the sum of the series: ∑(n(n+1)(n+2)) for n from 1 to 20",
        "expected_answer": "1830",
        "topic": "Algebra",
        "subtopic": "Series",
        "difficulty": "Hard"
    },
    
    # Coordinate Geometry
    {
        "id": "GEOM-1",
        "question": "Find the distance between the points (3, 4) and (-2, 8).",
        "expected_answer": "√41",
        "topic": "Coordinate Geometry",
        "subtopic": "Distance Formula",
        "difficulty": "Easy"
    },
    {
        "id": "GEOM-2",
        "question": "Find the equation of the circle passing through the origin and having its center at (2, 3).",
        "expected_answer": "x^2 + y^2 - 4x - 6y = 0",
        "topic": "Coordinate Geometry",
        "subtopic": "Circle",
        "difficulty": "Medium"
    },
    {
        "id": "GEOM-3",
        "question": "Find the area of the triangle formed by the points (1, 2), (3, 4), and (5, 0).",
        "expected_answer": "8",
        "topic": "Coordinate Geometry",
        "subtopic": "Area",
        "difficulty": "Medium"
    },
    
    # Trigonometry
    {
        "id": "TRIG-1",
        "question": "Prove that: sin(A+B)sin(A-B) = sin^2(A) - sin^2(B)",
        "expected_answer": "Proof using sin(A+B)sin(A-B) = sin^2(A) - sin^2(B)",
        "topic": "Trigonometry",
        "subtopic": "Identities",
        "difficulty": "Medium"
    },
    {
        "id": "TRIG-2",
        "question": "If tan A = 3/4 and tan B = 5/12, find tan(A+B).",
        "expected_answer": "56/33",
        "topic": "Trigonometry",
        "subtopic": "Addition Formulas",
        "difficulty": "Medium"
    },
    
    # Differential Equations
    {
        "id": "DE-1",
        "question": "Solve the differential equation: dy/dx + y = 2",
        "expected_answer": "y = Ce^(-x) + 2",
        "topic": "Differential Equations",
        "subtopic": "First Order",
        "difficulty": "Medium"
    },
    {
        "id": "DE-2",
        "question": "Solve the differential equation: d^2y/dx^2 - 4dy/dx + 4y = 0",
        "expected_answer": "y = (C1 + C2x)e^(2x)",
        "topic": "Differential Equations",
        "subtopic": "Second Order",
        "difficulty": "Hard"
    },
    
    # Vectors
    {
        "id": "VEC-1",
        "question": "If a = 2i + 3j - k and b = i - j + 2k, find a × b.",
        "expected_answer": "5i + 5j + 5k",
        "topic": "Vectors",
        "subtopic": "Cross Product",
        "difficulty": "Medium"
    },
    {
        "id": "VEC-2",
        "question": "Find the angle between the vectors a = i + j + k and b = i + j - k.",
        "expected_answer": "cos^(-1)(1/3)",
        "topic": "Vectors",
        "subtopic": "Dot Product",
        "difficulty": "Medium"
    },
    
    # Additional JEE Problems
    
    # More Calculus
    {
        "id": "CALC-4",
        "question": "If y = sin(ln(x)), find dy/dx.",
        "expected_answer": "cos(ln(x))/x",
        "topic": "Calculus",
        "subtopic": "Differentiation",
        "difficulty": "Medium"
    },
    {
        "id": "CALC-5",
        "question": "Evaluate the definite integral: ∫(0 to π/2) x sin(x) dx",
        "expected_answer": "1",
        "topic": "Calculus",
        "subtopic": "Definite Integration",
        "difficulty": "Medium"
    },
    {
        "id": "CALC-6",
        "question": "Find the area enclosed by the curve y = |x|, y = x² and x = 1.",
        "expected_answer": "1/3",
        "topic": "Calculus",
        "subtopic": "Integration Applications",
        "difficulty": "Hard"
    },
    
    # More Algebra
    {
        "id": "ALG-4",
        "question": "If a, b, c are in GP and a + b + c = 14, a·b·c = 64, find the value of b.",
        "expected_answer": "4",
        "topic": "Algebra",
        "subtopic": "Geometric Progression",
        "difficulty": "Hard"
    },
    {
        "id": "ALG-5",
        "question": "Solve the system of equations: log(x) + log(y) = 1, log(x·y) = 4",
        "expected_answer": "x = 10, y = 1 or x = 1, y = 10",
        "topic": "Algebra",
        "subtopic": "Logarithmic Equations",
        "difficulty": "Medium"
    },
    
    # More Trigonometry
    {
        "id": "TRIG-3",
        "question": "If sin θ + sin² θ = 1, find the value of cos⁴ θ - cos² θ.",
        "expected_answer": "0",
        "topic": "Trigonometry",
        "subtopic": "Trigonometric Equations",
        "difficulty": "Medium"
    },
    {
        "id": "TRIG-4",
        "question": "Prove that: (1+tan²A)(1+tan²B)(1+tan²C) = (tan A + tan B + tan C - tan A·tan B·tan C)² where A + B + C = π",
        "expected_answer": "Proof using the relation tan(A+B+C) = tan(π) = 0",
        "topic": "Trigonometry",
        "subtopic": "Advanced Identities",
        "difficulty": "Hard"
    },
    
    # Probability and Statistics
    {
        "id": "PROB-1",
        "question": "A fair coin is tossed 6 times. What is the probability of getting exactly 4 heads?",
        "expected_answer": "15/64",
        "topic": "Probability",
        "subtopic": "Binomial Distribution",
        "difficulty": "Easy"
    },
    {
        "id": "PROB-2",
        "question": "Three dice are rolled. What is the probability that the sum of the numbers is 10?",
        "expected_answer": "1/8",
        "topic": "Probability",
        "subtopic": "Dice Problems",
        "difficulty": "Medium"
    },
    
    # Complex Numbers
    {
        "id": "COMPLEX-1",
        "question": "If z = 1+i, find the value of z^6.",
        "expected_answer": "8i",
        "topic": "Complex Numbers",
        "subtopic": "Powers of Complex Numbers",
        "difficulty": "Medium"
    },
    {
        "id": "COMPLEX-2",
        "question": "Find all the values of z satisfying z^4 + 16 = 0.",
        "expected_answer": "z = ±2i, ±2i",
        "topic": "Complex Numbers",
        "subtopic": "Complex Roots",
        "difficulty": "Medium"
    },
    
    # 3D Geometry
    {
        "id": "3DGEOM-1",
        "question": "Find the equation of the plane passing through the point (1, 2, 3) and perpendicular to the vector 2i - j + 2k.",
        "expected_answer": "2x - y + 2z = 9",
        "topic": "3D Geometry",
        "subtopic": "Planes",
        "difficulty": "Medium"
    },
    {
        "id": "3DGEOM-2",
        "question": "Find the shortest distance between the lines r = (1,2,3) + t(2,1,1) and r = (3,1,2) + s(1,2,3).",
        "expected_answer": "√6/7",
        "topic": "3D Geometry",
        "subtopic": "Lines in 3D",
        "difficulty": "Hard"
    }
]

def evaluate_response(problem, response):
    """
    Evaluate the math agent's response to a JEE problem.
    
    Args:
        problem (dict): The problem dictionary
        response (str): The math agent's response
        
    Returns:
        dict: Evaluation results including correctness, confidence, etc.
    """
    # Basic evaluation (this would ideally be more sophisticated with NLP/LLM evaluation)
    expected_keywords = problem["expected_answer"].lower().split()
    response_lower = response.lower()
    
    # Check if key parts of the expected answer are in the response
    keyword_matches = sum(1 for keyword in expected_keywords if keyword in response_lower)
    keyword_percentage = keyword_matches / len(expected_keywords) if expected_keywords else 0
    
    # Determine if the answer appears to be correct
    correct = keyword_percentage > 0.7
    
    # Calculate confidence level based on keywords and response length
    confidence = min(1.0, keyword_percentage * 0.8 + min(1.0, len(response) / 300) * 0.2)
    
    # Calculate response time (placeholder - actual time is measured during processing)
    response_time = 0
    
    return {
        "problem_id": problem["id"],
        "correct": correct,
        "confidence": confidence,
        "response_time": response_time,
        "keyword_match_percentage": keyword_percentage
    }

def run_benchmark(problems=None, output_file=None, max_problems=None, generate_report=True):
    """
    Run the JEE benchmarking against the math agent.
    
    Args:
        problems (list): List of problem dictionaries to test
        output_file (str): Path to save the results JSON
        max_problems (int): Maximum number of problems to test
        generate_report (bool): Whether to generate a HTML report
        
    Returns:
        dict: Benchmark results
    """
    if problems is None:
        problems = JEE_PROBLEMS
    
    if max_problems is not None:
        problems = problems[:max_problems]
    
    # Initialize results
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_problems": len(problems),
        "problems_tested": 0,
        "correct_answers": 0,
        "avg_confidence": 0,
        "avg_response_time": 0,
        "problem_results": []
    }
    
    logger.info(f"Starting JEE benchmarking with {len(problems)} problems")
    
    # Process each problem
    for problem in tqdm(problems, desc="Testing JEE problems"):
        try:
            # Process the question
            start_time = time.time()
            response = process_query(problem["question"])
            end_time = time.time()
            response_time = end_time - start_time
            
            # Evaluate the response
            evaluation = evaluate_response(problem, response)
            evaluation["response_time"] = response_time
            evaluation["response"] = response
            
            # Add to results
            problem_result = {
                **problem,
                **evaluation,
                "response": response
            }
            results["problem_results"].append(problem_result)
            
            # Update summary statistics
            results["problems_tested"] += 1
            if evaluation["correct"]:
                results["correct_answers"] += 1
            
            # Log progress
            logger.info(f"Processed problem {problem['id']}: {'Correct' if evaluation['correct'] else 'Incorrect'} (Confidence: {evaluation['confidence']:.2f})")
            
        except Exception as e:
            logger.error(f"Error processing problem {problem['id']}: {e}")
            problem_result = {
                **problem,
                "error": str(e),
                "correct": False,
                "confidence": 0,
                "response_time": 0,
                "response": f"ERROR: {str(e)}"
            }
            results["problem_results"].append(problem_result)
            results["problems_tested"] += 1
    
    # Calculate final statistics
    if results["problems_tested"] > 0:
        results["accuracy"] = results["correct_answers"] / results["problems_tested"]
        results["avg_confidence"] = sum(r["confidence"] for r in results["problem_results"]) / results["problems_tested"]
        results["avg_response_time"] = sum(r["response_time"] for r in results["problem_results"]) / results["problems_tested"]
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    # Generate report
    if generate_report:
        report_file = output_file.replace('.json', '.html') if output_file else 'jee_benchmark_report.html'
        generate_html_report(results, report_file)
        logger.info(f"Report generated at {report_file}")
    
    return results

def generate_plots(results, output_dir="benchmark_plots"):
    """
    Generate plots for the benchmark results.
    
    Args:
        results (dict): Benchmark results
        output_dir (str): Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert problem results to DataFrame
    df = pd.DataFrame(results["problem_results"])
    
    # Plot 1: Accuracy by topic
    plt.figure(figsize=(10, 6))
    topic_accuracy = df.groupby('topic')['correct'].mean()
    topic_accuracy.plot(kind='bar')
    plt.title('Accuracy by Topic')
    plt.ylabel('Accuracy')
    plt.xlabel('Topic')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_topic.png'))
    
    # Plot 2: Accuracy by difficulty
    plt.figure(figsize=(10, 6))
    difficulty_accuracy = df.groupby('difficulty')['correct'].mean()
    difficulty_accuracy.plot(kind='bar')
    plt.title('Accuracy by Difficulty')
    plt.ylabel('Accuracy')
    plt.xlabel('Difficulty')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_difficulty.png'))
    
    # Plot 3: Response time by topic
    plt.figure(figsize=(10, 6))
    topic_time = df.groupby('topic')['response_time'].mean()
    topic_time.plot(kind='bar')
    plt.title('Average Response Time by Topic')
    plt.ylabel('Response Time (seconds)')
    plt.xlabel('Topic')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'response_time_by_topic.png'))
    
    # Plot 4: Confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['confidence'], bins=10)
    plt.title('Confidence Score Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
    
    return os.path.abspath(output_dir)

def generate_html_report(results, output_file="jee_benchmark_report.html"):
    """
    Generate a HTML report from the benchmark results.
    
    Args:
        results (dict): Benchmark results
        output_file (str): Path to save the HTML report
    """
    # Generate plots
    plots_dir = generate_plots(results)
    
    # Prepare the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>JEE Benchmark Results</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            .summary-stats {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 15px; }}
            .stat-card {{ background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); flex: 1; min-width: 200px; }}
            .stat-value {{ font-size: 24px; font-weight: bold; margin-top: 10px; color: #3498db; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .correct {{ color: green; }}
            .incorrect {{ color: red; }}
            .plots {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px; }}
            .plot {{ flex: 1; min-width: 500px; margin-bottom: 20px; }}
            .plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            .response {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px; white-space: pre-wrap; }}
            .difficulty-easy {{ background-color: #d4edda; }}
            .difficulty-medium {{ background-color: #fff3cd; }}
            .difficulty-hard {{ background-color: #f8d7da; }}
            .progress-container {{ width: 100%; background-color: #f3f3f3; border-radius: 5px; }}
            .progress-bar {{ height: 30px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>JEE Benchmarking Results for Math Agent</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="summary-stats">
                <div class="stat-card">
                    <h3>Accuracy</h3>
                    <div class="stat-value">{results['accuracy']*100:.1f}%</div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {results['accuracy']*100}%; background-color: {('#ff6b6b' if results['accuracy'] < 0.6 else '#feca57' if results['accuracy'] < 0.8 else '#1dd1a1')};">&nbsp;</div>
                    </div>
                </div>
                <div class="stat-card">
                    <h3>Problems Tested</h3>
                    <div class="stat-value">{results['problems_tested']}</div>
                </div>
                <div class="stat-card">
                    <h3>Avg. Confidence</h3>
                    <div class="stat-value">{results['avg_confidence']*100:.1f}%</div>
                </div>
                <div class="stat-card">
                    <h3>Avg. Response Time</h3>
                    <div class="stat-value">{results['avg_response_time']:.2f}s</div>
                </div>
            </div>
        </div>
        
        <h2>Performance Visualization</h2>
        <div class="plots">
            <div class="plot">
                <h3>Accuracy by Topic</h3>
                <img src="benchmark_plots/accuracy_by_topic.png" alt="Accuracy by Topic">
            </div>
            <div class="plot">
                <h3>Accuracy by Difficulty</h3>
                <img src="benchmark_plots/accuracy_by_difficulty.png" alt="Accuracy by Difficulty">
            </div>
            <div class="plot">
                <h3>Average Response Time by Topic</h3>
                <img src="benchmark_plots/response_time_by_topic.png" alt="Response Time by Topic">
            </div>
            <div class="plot">
                <h3>Confidence Score Distribution</h3>
                <img src="benchmark_plots/confidence_distribution.png" alt="Confidence Distribution">
            </div>
        </div>
        
        <h2>Detailed Results</h2>
        <table>
            <tr>
                <th>ID</th>
                <th>Topic</th>
                <th>Difficulty</th>
                <th>Question</th>
                <th>Expected Answer</th>
                <th>Correct</th>
                <th>Confidence</th>
                <th>Response Time</th>
                <th>Details</th>
            </tr>
    """
    
    # Add rows for each problem
    for result in results["problem_results"]:
        difficulty_class = f"difficulty-{result['difficulty'].lower()}"
        html_content += f"""
            <tr class="{difficulty_class}">
                <td>{result['id']}</td>
                <td>{result['topic']}</td>
                <td>{result['difficulty']}</td>
                <td>{result['question']}</td>
                <td>{result['expected_answer']}</td>
                <td class="{'correct' if result['correct'] else 'incorrect'}">{result['correct']}</td>
                <td>{result['confidence']*100:.1f}%</td>
                <td>{result['response_time']:.2f}s</td>
                <td>
                    <details>
                        <summary>View Response</summary>
                        <div class="response">{result['response']}</div>
                    </details>
                </td>
            </tr>
        """
    
    # Complete the HTML
    html_content += """
        </table>
        
        <h2>Conclusion</h2>
        <p>This report shows the performance of the Math Agent on JEE-level problems across different topics and difficulty levels. 
        The results can be used to identify areas of strength and weakness in the agent's capabilities.</p>
        
        <footer>
            <p>Math Agent JEE Benchmark - © 2024</p>
        </footer>
    </body>
    </html>
    """
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_file

def main():
    """Main function to run the JEE benchmark from command line"""
    parser = argparse.ArgumentParser(description='JEE Benchmarking Tool for Math Agent')
    parser.add_argument('--output', '-o', type=str, default='jee_benchmark_results.json',
                        help='Output file for benchmark results')
    parser.add_argument('--max-problems', '-m', type=int, default=None,
                        help='Maximum number of problems to test')
    parser.add_argument('--no-report', action='store_true',
                        help='Disable HTML report generation')
    
    args = parser.parse_args()
    
    results = run_benchmark(
        output_file=args.output,
        max_problems=args.max_problems,
        generate_report=not args.no_report
    )
    
    # Print summary
    print("\nJEE Benchmark Summary:")
    print(f"Problems Tested: {results['problems_tested']}")
    print(f"Accuracy: {results['accuracy']*100:.1f}%")
    print(f"Average Confidence: {results['avg_confidence']*100:.1f}%")
    print(f"Average Response Time: {results['avg_response_time']:.2f} seconds")
    
    if not args.no_report:
        report_file = args.output.replace('.json', '.html')
        print(f"\nDetailed report saved to: {report_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 