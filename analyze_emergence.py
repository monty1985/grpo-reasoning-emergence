"""
Analyze Reasoning Emergence from Training Logs
"""

import os
import json
import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_logs(log_dir: str):
    """Load completion logs"""
    pattern = os.path.join(log_dir, "completions", "step_*.json")
    files = sorted(glob.glob(pattern))
    
    logs = []
    for f in files:
        with open(f) as file:
            logs.append(json.load(file))
    
    return logs


def load_metrics(log_dir: str):
    """Load training metrics"""
    path = os.path.join(log_dir, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def analyze_patterns(logs):
    """Analyze reasoning pattern emergence"""
    
    steps = []
    accuracy = []
    avg_length = []
    patterns = defaultdict(list)
    
    for log in logs:
        step = log["step"]
        samples = log["samples"]
        
        steps.append(step)
        
        # Accuracy
        rewards = [s["reward"] for s in samples]
        correct = sum(1 for r in rewards if r == 1.0)
        accuracy.append(correct / len(samples) if samples else 0)
        
        # Length
        lengths = [len(s["completion"]) for s in samples]
        avg_length.append(sum(lengths) / len(lengths) if lengths else 0)
        
        # Patterns
        pattern_counts = defaultdict(int)
        for sample in samples:
            comp = sample["completion"].lower()
            
            if any(p in comp for p in ['first', 'then', 'step']):
                pattern_counts["step_by_step"] += 1
            if any(p in comp for p in ['check', 'verify']):
                pattern_counts["verification"] += 1
            if any(p in comp for p in ['try again', 'instead']):
                pattern_counts["backtracking"] += 1
            if "<think>" in sample["completion"]:
                pattern_counts["uses_think_tags"] += 1
        
        for k in ["step_by_step", "verification", "backtracking", "uses_think_tags"]:
            patterns[k].append(pattern_counts[k] / len(samples) if samples else 0)
    
    return {
        "steps": steps,
        "accuracy": accuracy,
        "avg_length": avg_length,
        "patterns": dict(patterns)
    }


def plot_emergence(analysis, output_path="emergence_analysis.png"):
    """Create visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    steps = analysis["steps"]
    
    # Accuracy
    ax = axes[0, 0]
    ax.plot(steps, analysis["accuracy"], 'b-', linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Over Training")
    ax.grid(True, alpha=0.3)
    
    # Response length
    ax = axes[0, 1]
    ax.plot(steps, analysis["avg_length"], 'g-', linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Avg Length (chars)")
    ax.set_title("Response Length (Longer = More Reasoning)")
    ax.grid(True, alpha=0.3)
    
    # Patterns
    ax = axes[1, 0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (pattern, values) in enumerate(analysis["patterns"].items()):
        ax.plot(steps, values, label=pattern, color=colors[i % len(colors)], linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Frequency")
    ax.set_title("Reasoning Pattern Emergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Metrics
    ax = axes[1, 1]
    metrics = load_metrics("./logs")
    if metrics:
        m_steps = [m["step"] for m in metrics]
        m_loss = [m["loss"] for m in metrics]
        ax.plot(m_steps, m_loss, 'r-', linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.show()


def print_summary(analysis):
    """Print emergence summary"""
    
    print("\n" + "=" * 60)
    print("REASONING EMERGENCE SUMMARY")
    print("=" * 60)
    
    if not analysis["steps"]:
        print("No data found.")
        return
    
    print(f"\n{'Metric':<25} {'Start':<12} {'End':<12} {'Change':<12}")
    print("-" * 61)
    
    # Accuracy
    start_acc = analysis["accuracy"][0]
    end_acc = analysis["accuracy"][-1]
    print(f"{'Accuracy':<25} {start_acc:<12.1%} {end_acc:<12.1%} {end_acc - start_acc:+.1%}")
    
    # Length
    start_len = analysis["avg_length"][0]
    end_len = analysis["avg_length"][-1]
    print(f"{'Avg Response Length':<25} {start_len:<12.0f} {end_len:<12.0f} {end_len - start_len:+.0f}")
    
    # Patterns
    print("\nReasoning Patterns:")
    for pattern, values in analysis["patterns"].items():
        start = values[0] if values else 0
        end = values[-1] if values else 0
        print(f"  {pattern:<23} {start:<12.1%} {end:<12.1%} {end - start:+.1%}")
    
    # Insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    
    if end_acc > start_acc + 0.1:
        print("✓ Significant accuracy improvement")
    if end_len > start_len * 1.5:
        print("✓ Responses became longer (more reasoning)")
    if analysis["patterns"].get("step_by_step", [0, 0])[-1] > analysis["patterns"].get("step_by_step", [0])[0] + 0.1:
        print("✓ Step-by-step reasoning emerged")
    if analysis["patterns"].get("verification", [0, 0])[-1] > 0.05:
        print("✓ Self-verification behavior emerged")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="./logs")
    args = parser.parse_args()
    
    print("Loading logs...")
    logs = load_logs(args.log_dir)
    
    if not logs:
        print("No logs found. Run training first.")
        exit(1)
    
    print(f"Found {len(logs)} log files")
    
    analysis = analyze_patterns(logs)
    print_summary(analysis)
    plot_emergence(analysis)