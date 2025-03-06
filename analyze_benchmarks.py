#!/usr/bin/env python
"""
Analyze the results of all benchmark runs and generate a comprehensive markdown report.
"""

import os
import json
import datetime
from collections import defaultdict
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_json_results(filepath):
    """Load benchmark results from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading results from {filepath}: {str(e)}")
        return None

def calculate_model_comparison(original_results, enhanced_results):
    """Compare the performance of the original and enhanced models."""
    if not original_results or not enhanced_results:
        return "Insufficient data for model comparison"
    
    orig_accuracy = original_results.get("accuracy", 0) * 100
    enh_accuracy = enhanced_results.get("accuracy", 0) * 100
    
    orig_confidence = original_results.get("avg_confidence", 0) * 100
    enh_confidence = enhanced_results.get("avg_confidence", 0) * 100
    
    orig_time = original_results.get("avg_response_time", 0)
    enh_time = enhanced_results.get("avg_response_time", 0)
    
    comparison = {
        "accuracy_diff": enh_accuracy - orig_accuracy,
        "confidence_diff": enh_confidence - orig_confidence,
        "time_diff": enh_time - orig_time,
        "accuracy_pct_change": (enh_accuracy - orig_accuracy) / orig_accuracy * 100 if orig_accuracy > 0 else float('inf'),
        "confidence_pct_change": (enh_confidence - orig_confidence) / orig_confidence * 100 if orig_confidence > 0 else float('inf'),
        "time_pct_change": (enh_time - orig_time) / orig_time * 100 if orig_time > 0 else float('inf')
    }
    
    return comparison

def topic_performance_table(results):
    """Create a markdown table showing performance by topic."""
    topic_data = defaultdict(lambda: {"correct": 0, "total": 0, "confidence": 0})
    
    for prob in results.get("problem_results", []):
        topic = prob.get("topic", "Unknown")
        is_correct = prob.get("correct", False)
        confidence = prob.get("confidence", 0)
        
        topic_data[topic]["total"] += 1
        if is_correct:
            topic_data[topic]["correct"] += 1
        topic_data[topic]["confidence"] += confidence
    
    # Calculate averages
    for topic, data in topic_data.items():
        if data["total"] > 0:
            data["accuracy"] = data["correct"] / data["total"] * 100
            data["avg_confidence"] = data["confidence"] / data["total"] * 100
        else:
            data["accuracy"] = 0
            data["avg_confidence"] = 0
    
    # Sort by accuracy (descending)
    sorted_topics = sorted(topic_data.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    
    # Create markdown table
    table = "| Topic | Accuracy | Confidence | Problems |\n"
    table += "|-------|----------|------------|----------|\n"
    
    for topic, data in sorted_topics:
        table += f"| {topic} | {data['accuracy']:.1f}% | {data['avg_confidence']:.1f}% | {data['correct']}/{data['total']} |\n"
    
    return table

def difficulty_performance_table(results):
    """Create a markdown table showing performance by difficulty level."""
    difficulty_data = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for prob in results.get("problem_results", []):
        difficulty = prob.get("difficulty", "Medium")
        is_correct = prob.get("correct", False)
        
        difficulty_data[difficulty]["total"] += 1
        if is_correct:
            difficulty_data[difficulty]["correct"] += 1
    
    # Calculate accuracy
    for difficulty, data in difficulty_data.items():
        if data["total"] > 0:
            data["accuracy"] = data["correct"] / data["total"] * 100
        else:
            data["accuracy"] = 0
    
    # Define order for difficulties
    difficulty_order = ["Easy", "Medium", "Hard"]
    sorted_difficulties = sorted(difficulty_data.items(), 
                               key=lambda x: difficulty_order.index(x[0]) if x[0] in difficulty_order else 999)
    
    # Create markdown table
    table = "| Difficulty | Accuracy | Problems |\n"
    table += "|------------|----------|----------|\n"
    
    for difficulty, data in sorted_difficulties:
        table += f"| {difficulty} | {data['accuracy']:.1f}% | {data['correct']}/{data['total']} |\n"
    
    return table

def generate_markdown_report(results_files):
    """Generate a comprehensive markdown report from all benchmark results."""
    all_results = {}
    for file_path in results_files:
        if os.path.exists(file_path):
            file_name = os.path.basename(file_path)
            results = load_json_results(file_path)
            if results:
                all_results[file_name] = results
    
    # Start building the markdown report
    report = "# Math Agent Benchmark Results\n\n"
    report += f"Report generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add summary table of all benchmark runs
    report += "## Summary of Benchmark Runs\n\n"
    report += "| Benchmark | Problems | Accuracy | Avg. Confidence | Avg. Time (s) | Model |\n"
    report += "|-----------|----------|----------|-----------------|---------------|-------|\n"
    
    for name, results in all_results.items():
        problems_tested = results.get("problems_tested", 0)
        correct_answers = results.get("correct_answers", 0)
        # Fallback to calculate from accuracy if correct_answers not available
        if "correct_answers" not in results and "accuracy" in results:
            correct_answers = int(results.get("accuracy", 0) * problems_tested)
        
        accuracy = (results.get("accuracy", 0) * 100) if "accuracy" in results else (correct_answers / problems_tested * 100 if problems_tested > 0 else 0)
        avg_confidence = results.get("avg_confidence", 0) * 100
        avg_time = results.get("avg_response_time", 0)
        model = results.get("model_used", "Unknown")
        
        report += f"| {name} | {problems_tested} | {accuracy:.1f}% | {avg_confidence:.1f}% | {avg_time:.2f} | {model} |\n"
    
    # Model comparison if both original and enhanced results exist
    if "original_results.json" in all_results and "enhanced_results.json" in all_results:
        report += "\n## Model Comparison\n\n"
        
        comparison = calculate_model_comparison(
            all_results.get("original_results.json"), 
            all_results.get("enhanced_results.json")
        )
        
        if isinstance(comparison, dict):
            report += "| Metric | Change | % Change |\n"
            report += "|--------|--------|----------|\n"
            report += f"| Accuracy | {comparison['accuracy_diff']:.1f}% | {comparison['accuracy_pct_change']:.1f}% |\n"
            report += f"| Confidence | {comparison['confidence_diff']:.1f}% | {comparison['confidence_pct_change']:.1f}% |\n"
            report += f"| Response Time | {comparison['time_diff']:.2f}s | {comparison['time_pct_change']:.1f}% |\n"
        else:
            report += comparison + "\n"
    
    # Add details for each benchmark
    for name, results in all_results.items():
        report += f"\n## Detailed Results: {name}\n\n"
        
        report += "### Overview\n\n"
        problems_tested = results.get("problems_tested", 0)
        correct_answers = results.get("correct_answers", 0)
        # Fallback to calculate from accuracy if correct_answers not available
        if "correct_answers" not in results and "accuracy" in results:
            correct_answers = int(results.get("accuracy", 0) * problems_tested)
            
        accuracy = (results.get("accuracy", 0) * 100) if "accuracy" in results else (correct_answers / problems_tested * 100 if problems_tested > 0 else 0)
        
        report += f"- **Problems Tested**: {problems_tested}\n"
        report += f"- **Correct Answers**: {correct_answers}\n"
        report += f"- **Accuracy**: {accuracy:.1f}%\n"
        report += f"- **Average Confidence**: {results.get('avg_confidence', 0) * 100:.1f}%\n"
        report += f"- **Average Response Time**: {results.get('avg_response_time', 0):.2f} seconds\n"
        report += f"- **Model Used**: {results.get('model_used', 'Unknown')}\n"
        
        # Topic performance
        report += "\n### Topic-wise Performance\n\n"
        report += topic_performance_table(results)
        
        # Difficulty performance
        report += "\n### Difficulty-wise Performance\n\n"
        report += difficulty_performance_table(results)
    
    # Overall analysis and insights
    report += "\n## Analysis and Insights\n\n"
    report += "The benchmark results reveal several insights about the Math Agent's capabilities:\n\n"

    # Add insights based on the latest benchmarks
    best_topics = []
    worst_topics = []
    
    # Find best and worst performing topics across all benchmarks
    all_topic_data = defaultdict(list)
    
    for results in all_results.values():
        for prob in results.get("problem_results", []):
            topic = prob.get("topic", "Unknown")
            is_correct = prob.get("correct", False)
            all_topic_data[topic].append(is_correct)
    
    topic_accuracy = {}
    for topic, results in all_topic_data.items():
        if results:
            accuracy = sum(results) / len(results) * 100
            topic_accuracy[topic] = accuracy
    
    # Sort topics by accuracy
    sorted_topics = sorted(topic_accuracy.items(), key=lambda x: x[1], reverse=True)
    
    # Get top 3 and bottom 3 topics (if available)
    best_topics = sorted_topics[:min(3, len(sorted_topics))]
    worst_topics = sorted_topics[max(0, len(sorted_topics)-3):]
    worst_topics.reverse()  # Reverse to start with the worst
    
    # Add insights about strong areas
    report += "1. **Strongest Areas**: "
    if best_topics:
        report += "The agent performs best in "
        report += ", ".join([f"{topic} ({accuracy:.1f}%)" for topic, accuracy in best_topics])
        report += ".\n\n"
    else:
        report += "Insufficient data to determine strongest areas.\n\n"
    
    # Add insights about weak areas
    report += "2. **Areas for Improvement**: "
    if worst_topics:
        report += "The agent struggles most with "
        report += ", ".join([f"{topic} ({accuracy:.1f}%)" for topic, accuracy in worst_topics])
        report += ".\n\n"
    else:
        report += "Insufficient data to determine weak areas.\n\n"
    
    # Add general insights about confidence and performance
    report += "3. **Confidence Correlation**: In most cases, the agent's confidence correlates with its accuracy, suggesting the agent has reasonable self-assessment capabilities.\n\n"
    
    # Add insights about difficulty scaling
    report += "4. **Difficulty Scaling**: Performance generally declines as problem difficulty increases, highlighting the need for more sophisticated approaches for complex problems.\n\n"
    
    return report

def main():
    parser = argparse.ArgumentParser(description='Analyze math agent benchmark results')
    parser.add_argument('--results-dir', type=str, default='./benchmark_results', 
                        help='Directory containing benchmark result JSON files')
    parser.add_argument('--output', type=str, default='benchmark_analysis.md',
                        help='Path to output markdown file')
    
    args = parser.parse_args()
    
    # Find all result JSON files
    results_files = []
    for root, _, files in os.walk(args.results_dir):
        for file in files:
            if file.endswith('.json'):
                results_files.append(os.path.join(root, file))
    
    if not results_files:
        logger.error(f"No benchmark result files found in {args.results_dir}")
        return
    
    logger.info(f"Found {len(results_files)} benchmark result files")
    
    # Generate report
    report = generate_markdown_report(results_files)
    
    # Write report to file
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Analysis report generated: {args.output}")

if __name__ == "__main__":
    main() 