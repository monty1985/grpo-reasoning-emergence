"""
Evaluation Script - Compare Base vs Trained Model
"""

import os
import json
import argparse
import re
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def load_model(path: str):
    """Load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def format_prompt(numbers, target, tokenizer):
    """Format as chat prompt"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer."},
        {"role": "user", "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags."}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, prompt: str, device) -> str:
    """Generate single response"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=384,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def verify_answer(completion: str, numbers: List[int], target: int) -> tuple:
    """Verify if answer is correct"""
    
    # Check format
    has_think = "<think>" in completion and "</think>" in completion
    has_answer = "<answer>" in completion and "</answer>" in completion
    
    if not has_answer:
        return False, has_think, None
    
    # Extract answer
    match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
    if not match:
        return False, has_think, None
    
    expr = match.group(1).strip()
    
    try:
        # Clean and evaluate
        expr_clean = expr.replace(' ', '')
        if not all(c in '0123456789+-*/()' for c in expr_clean):
            return False, has_think, expr
        
        result = eval(expr_clean, {"__builtins__": {}})
        is_correct = abs(result - target) < 1e-9
        
        return is_correct, has_think, expr
        
    except:
        return False, has_think, expr


def detect_patterns(completion: str) -> Dict[str, bool]:
    """Detect reasoning patterns"""
    c = completion.lower()
    
    return {
        "step_by_step": any(p in c for p in ['first', 'then', 'next', 'step']),
        "verification": any(p in c for p in ['check', 'verify', 'correct', 'equals']),
        "backtracking": any(p in c for p in ["doesn't work", 'try again', 'instead']),
        "exploration": any(p in c for p in ['let me try', 'what if', 'maybe']),
        "shows_work": bool(re.search(r'\d+\s*[+\-*/]\s*\d+\s*=', completion))
    }


def evaluate_model(
    model, 
    tokenizer, 
    dataset,
    num_samples: int = 100,
    device: str = "cuda"
) -> Dict:
    """Evaluate model on countdown problems"""
    
    results = {
        "correct": 0,
        "format_ok": 0,
        "total": 0,
        "patterns": {k: 0 for k in ["step_by_step", "verification", "backtracking", "exploration", "shows_work"]},
        "samples": []
    }
    
    indices = list(range(min(num_samples, len(dataset))))
    
    for idx in tqdm(indices, desc="Evaluating"):
        example = dataset[idx]
        prompt = format_prompt(example['nums'], example['target'], tokenizer)
        
        completion = generate(model, tokenizer, prompt, device)
        
        is_correct, has_format, expr = verify_answer(
            completion, example['nums'], example['target']
        )
        
        results["total"] += 1
        if is_correct:
            results["correct"] += 1
        if has_format:
            results["format_ok"] += 1
        
        # Patterns
        patterns = detect_patterns(completion)
        for k, v in patterns.items():
            if v:
                results["patterns"][k] += 1
        
        # Store samples
        if len(results["samples"]) < 10:
            results["samples"].append({
                "numbers": example['nums'],
                "target": example['target'],
                "completion": completion[:500],
                "correct": is_correct,
                "expression": expr
            })
    
    # Compute rates
    n = results["total"]
    results["accuracy"] = results["correct"] / n
    results["format_rate"] = results["format_ok"] / n
    for k in results["patterns"]:
        results["patterns"][k] /= n
    
    return results


def compare_models(base_path: str, trained_path: str, num_samples: int = 100):
    """Compare base vs trained model"""
    
    print("=" * 60)
    print("Model Comparison: Reasoning Emergence")
    print("=" * 60)
    
    # Load dataset
    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    dataset = dataset.shuffle(seed=42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Evaluate base
    print(f"\n1. Evaluating BASE: {base_path}")
    base_model, base_tok = load_model(base_path)
    base_results = evaluate_model(base_model, base_tok, dataset, num_samples, device)
    del base_model
    torch.cuda.empty_cache()
    
    # Evaluate trained
    print(f"\n2. Evaluating TRAINED: {trained_path}")
    trained_model, trained_tok = load_model(trained_path)
    trained_results = evaluate_model(trained_model, trained_tok, dataset, num_samples, device)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'Base':<12} {'Trained':<12} {'Change':<12}")
    print("-" * 61)
    
    print(f"{'Accuracy':<25} {base_results['accuracy']:<12.1%} {trained_results['accuracy']:<12.1%} {trained_results['accuracy'] - base_results['accuracy']:+.1%}")
    print(f"{'Format Rate':<25} {base_results['format_rate']:<12.1%} {trained_results['format_rate']:<12.1%} {trained_results['format_rate'] - base_results['format_rate']:+.1%}")
    
    print("\nReasoning Patterns:")
    for k in base_results["patterns"]:
        b = base_results["patterns"][k]
        t = trained_results["patterns"][k]
        print(f"  {k:<23} {b:<12.1%} {t:<12.1%} {t - b:+.1%}")
    
    # Sample outputs
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUTS")
    print("=" * 60)
    
    print("\n--- BASE MODEL ---")
    for s in base_results["samples"][:2]:
        print(f"\nNumbers: {s['numbers']}, Target: {s['target']}")
        print(f"Correct: {s['correct']}")
        print(f"Response: {s['completion'][:300]}...")
    
    print("\n--- TRAINED MODEL ---")
    for s in trained_results["samples"][:2]:
        print(f"\nNumbers: {s['numbers']}, Target: {s['target']}")
        print(f"Correct: {s['correct']}")
        print(f"Response: {s['completion'][:300]}...")
    
    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump({
            "base": base_results,
            "trained": trained_results
        }, f, indent=2)
    
    print("\n\nResults saved to evaluation_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--trained", default="./checkpoints/final")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    
    compare_models(args.base, args.trained, args.num_samples)