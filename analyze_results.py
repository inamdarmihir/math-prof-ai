import json

# Load the results file
with open('jee_expanded_benchmark_results.json', 'r') as f:
    data = json.load(f)

# Extract overall stats
total_problems = data['total_problems']
problems_tested = data['problems_tested']
correct_answers = data['correct_answers']
accuracy = correct_answers / problems_tested * 100
avg_confidence = data['avg_confidence'] * 100
avg_response_time = data['avg_response_time']

print(f"Overall Performance:")
print(f"Total problems: {total_problems}")
print(f"Problems tested: {problems_tested}")
print(f"Correct answers: {correct_answers}")
print(f"Accuracy: {accuracy:.1f}%")
print(f"Average confidence: {avg_confidence:.1f}%")
print(f"Average response time: {avg_response_time:.2f} seconds")
print()

# Analyze by topic
topics = {}
for result in data['problem_results']:
    topic = result['topic']
    if topic not in topics:
        topics[topic] = {'correct': 0, 'total': 0, 'confidence': 0}
    
    topics[topic]['total'] += 1
    if result['correct']:
        topics[topic]['correct'] += 1
    topics[topic]['confidence'] += result['confidence']

print("Topic-wise Performance:")
for topic, stats in topics.items():
    correct = stats['correct']
    total = stats['total']
    avg_topic_confidence = stats['confidence'] / total * 100
    print(f"{topic}: {correct}/{total} correct ({correct*100/total:.1f}%) - Avg Confidence: {avg_topic_confidence:.1f}%")

# Analyze by difficulty
difficulties = {}
for result in data['problem_results']:
    difficulty = result['difficulty']
    if difficulty not in difficulties:
        difficulties[difficulty] = {'correct': 0, 'total': 0}
    
    difficulties[difficulty]['total'] += 1
    if result['correct']:
        difficulties[difficulty]['correct'] += 1

print("\nDifficulty-wise Performance:")
for difficulty, stats in difficulties.items():
    correct = stats['correct']
    total = stats['total']
    print(f"{difficulty}: {correct}/{total} correct ({correct*100/total:.1f}%)") 