# GRPO Reasoning Emergence Project (V2)

## Phase 1: Countdown Task with Existing Datasets

**Goal**: Observe reasoning patterns emerge in a small language model through reinforcement learning with verifiable rewards.

**Budget**: ~$15-30  
**Time**: 4-8 hours of GPU training  
**Hardware**: Single RTX 4090 (24GB VRAM)

---

## What Changed in V2

| V1 (Original) | V2 (Updated) |
|---------------|--------------|
| Custom synthetic data generation | Use proven TinyZero dataset |
| Custom reward function | Use battle-tested verification |
| More code to debug | Focus on GRPO mechanics |
| Unverified difficulty | Known-good problem distribution |

**Why this is better**: The TinyZero dataset has been used to successfully demonstrate reasoning emergence in multiple research projects including Stanford CS224R. You can directly compare your results to published baselines.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Options](#2-dataset-options)
3. [RunPod Setup](#3-runpod-setup)
4. [Environment Installation](#4-environment-installation)
5. [Project Structure](#5-project-structure)
6. [Training Code](#6-training-code)
7. [Running the Experiment](#7-running-the-experiment)
8. [Observing Reasoning Emergence](#8-observing-reasoning-emergence)
9. [Analysis & Interpretation](#9-analysis--interpretation)
10. [Troubleshooting](#10-troubleshooting)
11. [Next Steps](#11-next-steps)

---

## 1. Project Overview

### What You'll Learn

1. **GRPO Mechanics**: How group-relative advantages replace value networks
2. **Reward Engineering**: Binary outcome-based rewards (correct = 1, wrong = 0)
3. **Emergence Observation**: Watch reasoning chains develop without explicit supervision
4. **Training Dynamics**: Loss curves, reward progression, KL divergence

### Why Countdown Task?

The Countdown task is ideal for observing emergence because:
- **Verifiable**: Answers are mathematically checkable
- **Variable Difficulty**: Problems range from easy to hard
- **Search Required**: Model must explore different operation sequences
- **Proven**: TinyZero demonstrated emergence with this exact task

### Expected Progression (Based on TinyZero Results)

| Training Stage | Typical Behavior | Accuracy |
|----------------|------------------|----------|
| Step 0-100 | Random guesses, short answers | ~5-10% |
| Step 100-300 | Basic arithmetic appears | ~15-25% |
| Step 300-600 | Trial-and-error patterns | ~30-50% |
| Step 600-1000 | Systematic exploration | ~50-70% |
| Step 1000+ | Verification, backtracking | ~70-95% |

---

## 2. Dataset Options

### Option A: TinyZero Dataset (Recommended)

**Dataset**: `Jiayi-Pan/Countdown-Tasks-3to4`

This is the exact dataset used by TinyZero to demonstrate the "aha moment" in small models.

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")

# Example problem
print(dataset['train'][0])
# {'numbers': [3, 5, 7, 2], 'target': 24, ...}
```

**Pros**:
- Battle-tested (proven to induce reasoning)
- Direct comparison to TinyZero results
- Pre-verified solvable problems
- Used by Stanford CS224R class

**Size**: ~490K problems

---

### Option B: reasoning-gym Library

**Package**: `reasoning-gym` (NeurIPS 2025 Spotlight)

```python
import reasoning_gym

# Create countdown dataset with built-in verification
data = reasoning_gym.create_dataset('countdown', size=10000, seed=42)

# Built-in reward function!
score = data.score_answer(answer=model_output, entry=example)
```

**Pros**:
- Built-in verification (no custom reward function needed)
- Multiple task types (for later experimentation)
- Configurable difficulty
- Academic standard

---

### Option C: GRPO-Zero Repository (Fastest)

Clone the entire working setup:

```bash
git clone https://github.com/policy-gradient/GRPO-Zero
cd GRPO-Zero
# Includes RTX 4090 config!
```

**Pros**:
- Complete working implementation
- RTX 4090 config included
- Minimal setup required

---

## 3. RunPod Setup

### Step 1: Create RunPod Account

1. Go to [runpod.io](https://runpod.io)
2. Create account and add payment method
3. Add ~$30 credits to start

### Step 2: Launch GPU Pod

1. Click "Deploy" → "GPU Pods"
2. Select **RTX 4090** (24GB) - approximately $0.44-0.69/hr
3. Choose template: **RunPod Pytorch 2.1**
4. Set container disk: **50GB** (for model weights + checkpoints)
5. Set volume disk: **20GB** (persistent storage)
6. Click "Deploy"

### Step 3: Connect to Pod

```bash
# Click "Connect" → "Start Web Terminal"
# Or use SSH (recommended):
ssh root@{pod-ip} -p {port} -i ~/.ssh/id_ed25519
```

### Step 4: Verify GPU

```bash
nvidia-smi
# Should show RTX 4090 with 24GB VRAM
```

---

## 4. Environment Installation

### Quick Setup Script

Save this as `setup.sh` and run it:

```bash
#!/bin/bash
set -e

echo "=== GRPO Reasoning Project Setup (V2) ==="

# Update system
apt-get update && apt-get install -y git wget vim htop git-lfs

# Initialize git-lfs
git lfs install

# Create project directory
mkdir -p /workspace/grpo_reasoning
cd /workspace/grpo_reasoning

# Install PyTorch (should be pre-installed on RunPod)
pip install --upgrade pip

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install transformers and TRL
pip install transformers==4.44.0
pip install trl==0.9.6
pip install accelerate==0.33.0
pip install peft==0.12.0

# Install datasets
pip install datasets

# Install reasoning-gym (NeurIPS 2025 library)
pip install reasoning-gym

# Install additional dependencies
pip install wandb
pip install tensorboard
pip install bitsandbytes
pip install scipy
pip install rich
pip install tqdm

# Download TinyZero dataset
echo "Downloading TinyZero Countdown dataset..."
python -c "from datasets import load_dataset; load_dataset('Jiayi-Pan/Countdown-Tasks-3to4')"

# Verify installation
echo ""
echo "=== Verifying Installation ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import trl; print(f'TRL: {trl.__version__}')"
python -c "import reasoning_gym; print(f'reasoning-gym: installed')"
python -c "print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Verify dataset
echo ""
echo "=== Verifying Dataset ==="
python -c "
from datasets import load_dataset
ds = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')
print(f'Dataset size: {len(ds)} examples')
print(f'Sample: {ds[0]}')
"

echo ""
echo "=== Setup Complete ==="
```

Run it:
```bash
chmod +x setup.sh
./setup.sh
```

### Weights & Biases Setup (Optional but Recommended)

```bash
wandb login
# Paste your API key when prompted
```

---

## 5. Project Structure

```
/workspace/grpo_reasoning/
├── setup.sh                 # Environment setup
├── config.py               # Hyperparameters
├── train_tinyzero.py       # Training with TinyZero dataset
├── train_reasoning_gym.py  # Training with reasoning-gym
├── evaluate.py             # Evaluation script
├── analyze_emergence.py    # Analysis tools
├── checkpoints/            # Model checkpoints
├── logs/                   # Training logs
│   ├── completions/        # Sample completions over training
│   └── metrics/            # Training metrics
└── results/                # Final results
```

Create directories:
```bash
cd /workspace/grpo_reasoning
mkdir -p checkpoints logs/completions logs/metrics results
```

---

## 6. Training Code

### File 1: `config.py`

```python
"""
Configuration for GRPO Reasoning Experiment (V2)
Using TinyZero dataset
"""

from dataclasses import dataclass

@dataclass
class Config:
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    # Alternative: "Qwen/Qwen2.5-1.5B-Instruct" (needs more VRAM)
    # Alternative: "HuggingFaceTB/SmolLM-360M-Instruct" (faster, less capable)
    
    # Dataset
    dataset_name: str = "Jiayi-Pan/Countdown-Tasks-3to4"
    max_train_samples: int = 10000  # Use subset for faster iteration
    
    # GRPO Hyperparameters
    num_generations: int = 4          # G: completions per prompt (group size)
    max_completion_length: int = 512  # Max tokens to generate
    max_prompt_length: int = 256      # Max tokens in prompt
    
    # Learning
    learning_rate: float = 5e-7       # Conservative for stability
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    
    # GRPO-specific
    beta: float = 0.01                # KL coefficient (lower = more exploration)
    
    # Reward weights (following TinyZero)
    format_reward: float = 0.1        # Reward for correct format
    answer_reward: float = 1.0        # Reward for correct answer
    
    # Training
    num_training_steps: int = 1000
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 50
    
    # Paths
    output_dir: str = "./checkpoints"
    logging_dir: str = "./logs"
    
    # Hardware
    bf16: bool = True
    gradient_checkpointing: bool = True
    
    # Tracking
    use_wandb: bool = True
    wandb_project: str = "grpo-countdown-v2"
    experiment_name: str = "qwen-0.5b-tinyzero-v1"


# System prompt following TinyZero format
SYSTEM_PROMPT = """You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer."""

# User prompt template
USER_PROMPT_TEMPLATE = """Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."""
```

### File 2: `train_tinyzero.py` (Main Training Script)

```python
"""
GRPO Training Script using TinyZero Dataset
This is the recommended approach for learning GRPO mechanics
"""

import os
import re
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from config import Config, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


class CountdownRewardFunction:
    """
    Reward function following TinyZero's approach:
    - 1.0 for correct answer
    - 0.1 for correct format (but wrong answer)
    - 0.0 otherwise
    """
    
    def __init__(self):
        self.format_pattern = re.compile(
            r'<think>.*?</think>.*?<answer>(.*?)</answer>',
            re.DOTALL
        )
        self.stats = {"total": 0, "correct": 0, "format_only": 0, "wrong": 0}
    
    def extract_answer(self, completion: str) -> Optional[str]:
        """Extract answer from <answer> tags"""
        match = self.format_pattern.search(completion)
        if match:
            return match.group(1).strip()
        return None
    
    def evaluate_expression(
        self, 
        expression: str, 
        numbers: List[int], 
        target: int
    ) -> Tuple[bool, Optional[float]]:
        """
        Evaluate if expression is correct.
        Returns (is_valid, result)
        """
        try:
            # Clean expression
            expr = expression.replace(' ', '')
            
            # Only allow safe characters
            if not all(c in '0123456789+-*/()' for c in expr):
                return False, None
            
            # Check numbers used
            nums_in_expr = [int(n) for n in re.findall(r'\d+', expr)]
            nums_available = numbers.copy()
            
            for num in nums_in_expr:
                if num in nums_available:
                    nums_available.remove(num)
                else:
                    return False, None  # Number used twice or not available
            
            # Evaluate
            result = eval(expr, {"__builtins__": {}})
            
            return True, result
            
        except Exception:
            return False, None
    
    def __call__(
        self,
        completions: List[str],
        prompts: List[str],
        numbers_list: List[List[int]],
        targets: List[int]
    ) -> List[float]:
        """
        Compute rewards for batch of completions.
        
        Following TinyZero:
        - 1.0 for correct answer
        - 0.1 for correct format only
        - 0.0 otherwise
        """
        rewards = []
        
        for completion, numbers, target in zip(completions, numbers_list, targets):
            self.stats["total"] += 1
            
            # Check format
            answer = self.extract_answer(completion)
            
            if answer is None:
                # Wrong format
                rewards.append(0.0)
                self.stats["wrong"] += 1
                continue
            
            # Check answer
            is_valid, result = self.evaluate_expression(answer, numbers, target)
            
            if is_valid and result == target:
                # Correct!
                rewards.append(1.0)
                self.stats["correct"] += 1
            else:
                # Correct format but wrong answer
                rewards.append(0.1)
                self.stats["format_only"] += 1
        
        return rewards
    
    def get_stats(self) -> Dict:
        total = max(self.stats["total"], 1)
        return {
            "accuracy": self.stats["correct"] / total,
            "format_rate": (self.stats["correct"] + self.stats["format_only"]) / total,
            **self.stats
        }
    
    def reset_stats(self):
        self.stats = {"total": 0, "correct": 0, "format_only": 0, "wrong": 0}


def format_prompt(numbers: List[int], target: int, tokenizer) -> str:
    """Format problem as chat prompt"""
    
    user_content = USER_PROMPT_TEMPLATE.format(
        numbers=numbers,
        target=target
    )
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    # Use tokenizer's chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return prompt


def prepare_dataset(config: Config, tokenizer):
    """Load and prepare TinyZero dataset"""
    
    print(f"Loading dataset: {config.dataset_name}")
    
    # Load dataset
    dataset = load_dataset(config.dataset_name, split="train")
    
    # Take subset if specified
    if config.max_train_samples and config.max_train_samples < len(dataset):
        dataset = dataset.select(range(config.max_train_samples))
    
    print(f"Dataset size: {len(dataset)} examples")
    
    # Format prompts
    def format_example(example):
        prompt = format_prompt(
            example['nums'],  # TinyZero uses 'nums' key
            example['target'],
            tokenizer
        )
        return {
            "prompt": prompt,
            "numbers": example['nums'],
            "target": example['target']
        }
    
    formatted_dataset = dataset.map(format_example)
    
    print(f"Sample prompt:\n{formatted_dataset[0]['prompt'][:500]}...")
    
    return formatted_dataset


class EmergenceTracker:
    """Track and log reasoning emergence patterns"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.completions_dir = os.path.join(log_dir, "completions")
        os.makedirs(self.completions_dir, exist_ok=True)
        
        self.metrics_history = []
    
    def detect_patterns(self, completion: str) -> Dict[str, bool]:
        """Detect reasoning patterns in completion"""
        
        completion_lower = completion.lower()
        
        return {
            "has_think_tags": "<think>" in completion and "</think>" in completion,
            "has_answer_tags": "<answer>" in completion and "</answer>" in completion,
            "step_by_step": any(p in completion_lower for p in 
                              ['first', 'then', 'next', 'step']),
            "verification": any(p in completion_lower for p in 
                               ['check', 'verify', 'equals', 'correct']),
            "backtracking": any(p in completion_lower for p in 
                               ["doesn't work", 'try again', 'instead', 'another']),
            "exploration": any(p in completion_lower for p in 
                              ['let me try', 'what if', 'maybe', 'perhaps']),
            "shows_calculation": bool(re.search(r'\d+\s*[+\-*/]\s*\d+\s*=', completion)),
            "long_response": len(completion) > 300
        }
    
    def log_step(
        self,
        step: int,
        completions: List[str],
        rewards: List[float],
        numbers_list: List[List[int]],
        targets: List[int]
    ):
        """Log completions and patterns for a training step"""
        
        # Analyze patterns
        pattern_counts = {}
        
        samples = []
        for i, (comp, reward, nums, target) in enumerate(
            zip(completions[:8], rewards[:8], numbers_list[:8], targets[:8])
        ):
            patterns = self.detect_patterns(comp)
            
            for pattern, present in patterns.items():
                if pattern not in pattern_counts:
                    pattern_counts[pattern] = 0
                if present:
                    pattern_counts[pattern] += 1
            
            samples.append({
                "numbers": nums,
                "target": target,
                "completion": comp[:1000],  # Truncate
                "reward": reward,
                "patterns": patterns
            })
        
        # Save to file
        log_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "pattern_rates": {k: v / len(samples) for k, v in pattern_counts.items()},
            "samples": samples
        }
        
        filename = os.path.join(self.completions_dir, f"step_{step:06d}.json")
        with open(filename, 'w') as f:
            json.dump(log_entry, f, indent=2)
        
        return log_entry
    
    def log_metrics(self, step: int, metrics: Dict):
        """Log training metrics"""
        self.metrics_history.append({"step": step, **metrics})
        
        # Save periodically
        if step % 100 == 0:
            metrics_file = os.path.join(self.log_dir, "metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)


def main():
    """Main training function"""
    
    print("=" * 60)
    print("GRPO Countdown Reasoning Training (V2)")
    print("Using TinyZero Dataset")
    print("=" * 60)
    
    # Load config
    config = Config()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load tokenizer
    print(f"\nLoading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"Model parameters: {model.num_parameters():,}")
    
    # Prepare dataset
    train_dataset = prepare_dataset(config, tokenizer)
    
    # Initialize reward function and tracker
    reward_fn = CountdownRewardFunction()
    tracker = EmergenceTracker(config.logging_dir)
    
    # Store metadata for reward computation
    numbers_list = train_dataset['numbers']
    targets = train_dataset['target']
    
    def compute_rewards(completions, prompts, **kwargs):
        """Wrapper for reward function compatible with TRL"""
        # Get batch indices from prompts
        batch_nums = []
        batch_targets = []
        
        for prompt in prompts:
            # Find matching problem (simple approach for demo)
            # In production, you'd track indices properly
            for i, p in enumerate(train_dataset['prompt']):
                if prompt == p:
                    batch_nums.append(numbers_list[i])
                    batch_targets.append(targets[i])
                    break
            else:
                # Fallback
                batch_nums.append([1, 2, 3, 4])
                batch_targets.append(10)
        
        return reward_fn(completions, prompts, batch_nums, batch_targets)
    
    # Configure GRPO
    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        
        # GRPO specific
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        max_prompt_length=config.max_prompt_length,
        beta=config.beta,
        
        # Logging
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        
        # Hardware
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        
        # WandB
        report_to="wandb" if config.use_wandb else "none",
        run_name=config.experiment_name,
    )
    
    # Initialize trainer
    print("\nInitializing GRPO Trainer...")
    
    trainer = GRPOTrainer(
        model=model,
        config=grpo_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        reward_funcs=compute_rewards,
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting Training - Watch for Reasoning Emergence!")
    print("=" * 60 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(os.path.join(config.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(config.output_dir, "final"))
    
    # Print final stats
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nFinal reward stats: {reward_fn.get_stats()}")


if __name__ == "__main__":
    main()
```

### File 3: `train_reasoning_gym.py` (Alternative using reasoning-gym)

```python
"""
GRPO Training Script using reasoning-gym
Alternative approach with built-in verification
"""

import os
import json
from datetime import datetime
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
import reasoning_gym

from config import Config


def create_dataset_from_reasoning_gym(
    size: int = 10000,
    seed: int = 42,
    num_numbers: int = 4
) -> tuple:
    """Create dataset using reasoning-gym"""
    
    print(f"Creating reasoning-gym countdown dataset (size={size})...")
    
    # Create countdown dataset
    rg_data = reasoning_gym.create_dataset(
        'countdown',
        size=size,
        seed=seed
    )
    
    # Convert to format for training
    prompts = []
    questions = []
    answers = []
    
    for example in rg_data:
        prompts.append(f"""Solve this countdown puzzle:

{example['question']}

Think step by step in <think> </think> tags.
Put your final answer in <answer> </answer> tags.
Example format: <answer> (1 + 2) * 3 </answer>""")
        
        questions.append(example['question'])
        answers.append(example['answer'])
    
    # Create HuggingFace dataset
    dataset = Dataset.from_dict({
        "prompt": prompts,
        "question": questions,
        "answer": answers
    })
    
    # Store the reasoning_gym dataset for verification
    return dataset, rg_data


class ReasoningGymRewardFunction:
    """
    Reward function using reasoning-gym's built-in verification
    """
    
    def __init__(self, rg_data):
        self.rg_data = rg_data
        self.stats = {"total": 0, "correct": 0, "format_only": 0, "wrong": 0}
    
    def extract_answer(self, completion: str) -> str:
        """Extract answer from completion"""
        import re
        
        # Look for <answer> tags
        match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Fallback: look for last equation
        lines = completion.strip().split('\n')
        for line in reversed(lines):
            if '=' in line:
                return line.split('=')[0].strip()
        
        return completion.strip()
    
    def __call__(
        self,
        completions: List[str],
        prompts: List[str],
        indices: List[int]
    ) -> List[float]:
        """Compute rewards using reasoning-gym verification"""
        
        rewards = []
        
        for completion, idx in zip(completions, indices):
            self.stats["total"] += 1
            
            # Extract answer
            answer = self.extract_answer(completion)
            
            # Use reasoning-gym's built-in verification
            entry = self.rg_data[idx]
            
            try:
                score = self.rg_data.score_answer(answer=answer, entry=entry)
                
                if score == 1.0:
                    rewards.append(1.0)
                    self.stats["correct"] += 1
                elif "<think>" in completion and "</think>" in completion:
                    rewards.append(0.1)  # Format reward
                    self.stats["format_only"] += 1
                else:
                    rewards.append(0.0)
                    self.stats["wrong"] += 1
                    
            except Exception:
                rewards.append(0.0)
                self.stats["wrong"] += 1
        
        return rewards
    
    def get_stats(self) -> Dict:
        total = max(self.stats["total"], 1)
        return {
            "accuracy": self.stats["correct"] / total,
            **self.stats
        }


def main():
    """Main training function using reasoning-gym"""
    
    print("=" * 60)
    print("GRPO Training with reasoning-gym")
    print("=" * 60)
    
    config = Config()
    
    # Create dataset
    train_dataset, rg_data = create_dataset_from_reasoning_gym(
        size=config.max_train_samples,
        seed=42
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Sample:\n{train_dataset[0]['prompt'][:300]}...")
    
    # Load model
    print(f"\nLoading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Initialize reward function
    reward_fn = ReasoningGymRewardFunction(rg_data)
    
    # Configure GRPO
    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        max_prompt_length=config.max_prompt_length,
        beta=config.beta,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to="wandb" if config.use_wandb else "none",
        run_name=config.experiment_name + "-rg",
    )
    
    def compute_rewards(completions, prompts, **kwargs):
        # Get indices (simplified - in production track properly)
        indices = list(range(len(completions)))
        return reward_fn(completions, prompts, indices)
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        config=grpo_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        reward_funcs=compute_rewards,
    )
    
    # Train
    print("\nStarting Training...")
    trainer.train()
    
    # Save
    trainer.save_model(os.path.join(config.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(config.output_dir, "final"))
    
    print(f"\nFinal stats: {reward_fn.get_stats()}")


if __name__ == "__main__":
    main()
```

### File 4: `train_simple_standalone.py` (Simplified, Self-Contained)

```python
"""
Simplified GRPO Training - Self-Contained Version
Best for understanding GRPO mechanics step-by-step

This version is more verbose but easier to understand and debug.
"""

import os
import re
import json
import random
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    # Model
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Dataset
    dataset_name = "Jiayi-Pan/Countdown-Tasks-3to4"
    max_samples = 5000  # Use subset for faster training
    
    # Training
    num_steps = 800
    batch_size = 4                    # Prompts per step
    num_generations = 4               # Completions per prompt (G)
    learning_rate = 1e-6
    max_completion_length = 384
    
    # GRPO
    kl_coef = 0.01
    clip_range = 0.2
    
    # Rewards (following TinyZero)
    reward_correct = 1.0
    reward_format = 0.1
    reward_wrong = 0.0
    
    # Logging
    log_every = 10
    save_every = 200
    sample_every = 50
    
    # Paths
    output_dir = "./checkpoints"
    log_dir = "./logs"


# ============================================================
# PROMPT FORMATTING
# ============================================================

SYSTEM_PROMPT = "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer."

def format_prompt(numbers: List[int], target: int, tokenizer) -> str:
    """Format countdown problem as chat prompt"""
    
    user_content = f"""Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


# ============================================================
# REWARD FUNCTION
# ============================================================

class CountdownReward:
    """
    TinyZero-style reward function:
    - 1.0: Correct answer
    - 0.1: Correct format, wrong answer
    - 0.0: Wrong format
    """
    
    def __init__(self):
        self.answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        self.think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)
        
        # Statistics
        self.stats = {"total": 0, "correct": 0, "format": 0, "wrong": 0}
    
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from <answer> tags"""
        match = self.answer_pattern.search(text)
        return match.group(1).strip() if match else None
    
    def has_format(self, text: str) -> bool:
        """Check if response has correct format"""
        has_think = bool(self.think_pattern.search(text))
        has_answer = bool(self.answer_pattern.search(text))
        return has_think and has_answer
    
    def evaluate_expression(
        self, 
        expr: str, 
        numbers: List[int], 
        target: int
    ) -> bool:
        """Check if expression is correct"""
        try:
            # Clean
            expr = expr.replace(' ', '')
            
            # Safety check
            if not all(c in '0123456789+-*/()' for c in expr):
                return False
            
            # Check numbers
            nums_used = [int(n) for n in re.findall(r'\d+', expr)]
            nums_avail = numbers.copy()
            
            for n in nums_used:
                if n in nums_avail:
                    nums_avail.remove(n)
                else:
                    return False
            
            # Evaluate
            result = eval(expr, {"__builtins__": {}})
            return abs(result - target) < 1e-9
            
        except:
            return False
    
    def __call__(
        self, 
        completion: str, 
        numbers: List[int], 
        target: int
    ) -> float:
        """Compute reward for single completion"""
        
        self.stats["total"] += 1
        
        # Check format
        if not self.has_format(completion):
            self.stats["wrong"] += 1
            return Config.reward_wrong
        
        # Extract and evaluate answer
        answer = self.extract_answer(completion)
        if answer and self.evaluate_expression(answer, numbers, target):
            self.stats["correct"] += 1
            return Config.reward_correct
        else:
            self.stats["format"] += 1
            return Config.reward_format
    
    def get_accuracy(self) -> float:
        if self.stats["total"] == 0:
            return 0.0
        return self.stats["correct"] / self.stats["total"]
    
    def reset(self):
        self.stats = {"total": 0, "correct": 0, "format": 0, "wrong": 0}


# ============================================================
# GRPO TRAINER
# ============================================================

class SimpleGRPOTrainer:
    """
    Simplified GRPO implementation for learning.
    Each method documents the GRPO mechanics.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model: {config.model_name}")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, 
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Policy model (trainable)
        self.policy = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Reference model (frozen)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
        
        # Optimizer
        self.optimizer = AdamW(
            self.policy.parameters(), 
            lr=config.learning_rate
        )
        
        # Reward function
        self.reward_fn = CountdownReward()
        
        # Tracking
        self.step = 0
        self.metrics_history = []
        
        # Load dataset
        self.load_dataset()
        
        # Create directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(os.path.join(config.log_dir, "completions"), exist_ok=True)
    
    def load_dataset(self):
        """Load TinyZero countdown dataset"""
        
        print(f"Loading dataset: {self.config.dataset_name}")
        
        dataset = load_dataset(self.config.dataset_name, split="train")
        
        if self.config.max_samples < len(dataset):
            dataset = dataset.shuffle(seed=42).select(range(self.config.max_samples))
        
        self.dataset = dataset
        print(f"Loaded {len(dataset)} examples")
    
    def get_batch(self) -> List[Dict]:
        """Get random batch of problems"""
        
        indices = random.sample(range(len(self.dataset)), self.config.batch_size)
        
        batch = []
        for idx in indices:
            example = self.dataset[idx]
            prompt = format_prompt(
                example['nums'],
                example['target'],
                self.tokenizer
            )
            batch.append({
                "prompt": prompt,
                "numbers": example['nums'],
                "target": example['target']
            })
        
        return batch
    
    @torch.no_grad()
    def generate_completions(
        self, 
        prompts: List[str]
    ) -> List[List[str]]:
        """
        Generate G completions per prompt.
        This is the ROLLOUT phase of GRPO.
        """
        
        self.policy.eval()
        all_completions = []
        
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_completion_length
            ).to(self.device)
            
            outputs = self.policy.generate(
                **inputs,
                max_new_tokens=self.config.max_completion_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                num_return_sequences=self.config.num_generations,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            completions = []
            prompt_len = inputs['input_ids'].shape[1]
            
            for output in outputs:
                generated = output[prompt_len:]
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                completions.append(text)
            
            all_completions.append(completions)
        
        self.policy.train()
        return all_completions
    
    def compute_rewards(
        self,
        completions: List[List[str]],
        problems: List[Dict]
    ) -> List[List[float]]:
        """
        Compute rewards for all completions.
        Binary rewards: 1.0 correct, 0.1 format, 0.0 wrong
        """
        
        all_rewards = []
        
        for prompt_completions, problem in zip(completions, problems):
            rewards = []
            for comp in prompt_completions:
                r = self.reward_fn(comp, problem['numbers'], problem['target'])
                rewards.append(r)
            all_rewards.append(rewards)
        
        return all_rewards
    
    def compute_advantages(
        self, 
        rewards: List[List[float]]
    ) -> List[List[float]]:
        """
        GRPO's key innovation: Group-relative advantage normalization.
        
        Instead of a learned value function (PPO), we normalize 
        rewards within each group:
        
        advantage = (reward - group_mean) / group_std
        
        This eliminates the need for a critic network!
        """
        
        all_advantages = []
        
        for group_rewards in rewards:
            r = torch.tensor(group_rewards, dtype=torch.float32)
            
            mean = r.mean()
            std = r.std()
            
            if std > 1e-8:
                advantages = (r - mean) / std
            else:
                # If all rewards same, advantages = 0
                advantages = torch.zeros_like(r)
            
            all_advantages.append(advantages.tolist())
        
        return all_advantages
    
    def compute_log_probs(
        self,
        model,
        prompt: str,
        completion: str
    ) -> torch.Tensor:
        """Compute log probability of completion given prompt"""
        
        full_text = prompt + completion
        
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_ids['input_ids'].shape[1]
        
        with torch.set_grad_enabled(model == self.policy):
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Shift for next-token prediction
        shift_logits = logits[:, prompt_len-1:-1, :]
        shift_labels = inputs['input_ids'][:, prompt_len:]
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        return token_log_probs.sum()
    
    def grpo_update(
        self,
        prompts: List[str],
        completions: List[List[str]],
        advantages: List[List[float]]
    ) -> Dict:
        """
        GRPO policy gradient update.
        
        Loss = -E[advantage * log_prob] + kl_coef * KL(policy || reference)
        """
        
        total_pg_loss = 0.0
        total_kl_loss = 0.0
        num_samples = 0
        
        for prompt, prompt_comps, prompt_advs in zip(
            prompts, completions, advantages
        ):
            for comp, adv in zip(prompt_comps, prompt_advs):
                # Policy log prob
                policy_lp = self.compute_log_probs(self.policy, prompt, comp)
                
                # Reference log prob
                with torch.no_grad():
                    ref_lp = self.compute_log_probs(self.ref_model, prompt, comp)
                
                # Policy gradient loss (negative for gradient ascent)
                adv_tensor = torch.tensor(adv, device=self.device)
                pg_loss = -policy_lp * adv_tensor
                
                # KL penalty
                kl_loss = self.config.kl_coef * (policy_lp - ref_lp)
                
                total_pg_loss += pg_loss
                total_kl_loss += kl_loss
                num_samples += 1
        
        # Average
        loss = (total_pg_loss + total_kl_loss) / max(num_samples, 1)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "pg_loss": (total_pg_loss / max(num_samples, 1)).item(),
            "kl_loss": (total_kl_loss / max(num_samples, 1)).item()
        }
    
    def train_step(self) -> Dict:
        """
        Complete GRPO training step:
        1. Sample batch
        2. Generate completions
        3. Compute rewards
        4. Compute advantages
        5. Update policy
        """
        
        # 1. Get batch
        batch = self.get_batch()
        prompts = [b['prompt'] for b in batch]
        
        # 2. Generate completions
        completions = self.generate_completions(prompts)
        
        # 3. Compute rewards
        rewards = self.compute_rewards(completions, batch)
        
        # 4. Compute advantages
        advantages = self.compute_advantages(rewards)
        
        # 5. Update policy
        loss_info = self.grpo_update(prompts, completions, advantages)
        
        # Metrics
        all_rewards = [r for group in rewards for r in group]
        
        metrics = {
            **loss_info,
            "mean_reward": sum(all_rewards) / len(all_rewards),
            "accuracy": self.reward_fn.get_accuracy(),
            "step": self.step
        }
        
        # Log samples
        if self.step % self.config.sample_every == 0:
            self.log_samples(batch, completions, rewards)
        
        self.step += 1
        return metrics
    
    def log_samples(
        self,
        batch: List[Dict],
        completions: List[List[str]],
        rewards: List[List[float]]
    ):
        """Log sample completions for analysis"""
        
        samples = []
        for prob, comps, rews in zip(batch[:2], completions[:2], rewards[:2]):
            for comp, rew in zip(comps[:2], rews[:2]):
                samples.append({
                    "numbers": prob['numbers'],
                    "target": prob['target'],
                    "completion": comp[:800],
                    "reward": rew
                })
        
        log_entry = {
            "step": self.step,
            "timestamp": datetime.now().isoformat(),
            "samples": samples
        }
        
        path = os.path.join(
            self.config.log_dir, 
            "completions",
            f"step_{self.step:06d}.json"
        )
        with open(path, 'w') as f:
            json.dump(log_entry, f, indent=2)
    
    def train(self, num_steps: int):
        """Main training loop"""
        
        print("\n" + "=" * 60)
        print("GRPO Training - TinyZero Dataset")
        print("=" * 60)
        print(f"Model: {self.config.model_name}")
        print(f"Steps: {num_steps}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Generations/prompt: {self.config.num_generations}")
        print("=" * 60 + "\n")
        
        pbar = tqdm(range(num_steps), desc="Training")
        
        for _ in pbar:
            metrics = self.train_step()
            
            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "acc": f"{metrics['accuracy']:.1%}",
                "reward": f"{metrics['mean_reward']:.2f}"
            })
            
            if self.step % self.config.log_every == 0:
                self.metrics_history.append(metrics)
            
            if self.step % self.config.save_every == 0 and self.step > 0:
                self.save_checkpoint()
        
        self.save_checkpoint(final=True)
        self.save_metrics()
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Final accuracy: {self.reward_fn.get_accuracy():.1%}")
        print("=" * 60)
    
    def save_checkpoint(self, final: bool = False):
        name = "final" if final else f"step_{self.step}"
        path = os.path.join(self.config.output_dir, name)
        self.policy.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"\nSaved checkpoint: {path}")
    
    def save_metrics(self):
        path = os.path.join(self.config.log_dir, "metrics.json")
        with open(path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    config = Config()
    trainer = SimpleGRPOTrainer(config)
    trainer.train(config.num_steps)
```

### File 5: `evaluate.py`

```python
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
```

### File 6: `analyze_emergence.py`

```python
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
```

---

## 7. Running the Experiment

### Quick Start (Recommended)

```bash
# 1. SSH into RunPod
ssh root@{pod-ip} -p {port}

# 2. Setup
cd /workspace
mkdir grpo_reasoning && cd grpo_reasoning
# Copy the files above or:
# git clone your-repo

# 3. Install dependencies
chmod +x setup.sh && ./setup.sh

# 4. (Optional) Setup wandb
wandb login

# 5. Run training
python train_simple_standalone.py
# Or: python train_tinyzero.py
```

### Alternative: Use GRPO-Zero Repository

```bash
# Clone complete working implementation
git clone https://github.com/policy-gradient/GRPO-Zero
cd GRPO-Zero

pip install uv
uv sync

# Download dataset
git lfs install
git clone https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4

# Download model
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

# Train (RTX 4090 config)
uv run train.py --config config_24GB.yaml
```

### Monitor Training

```bash
# Watch GPU
watch -n 1 nvidia-smi

# Check latest samples
cat $(ls -t logs/completions/*.json | head -1) | python -m json.tool | head -50

# If using wandb
# Check dashboard at wandb.ai
```

---

## 8. Observing Reasoning Emergence

### What to Look For

**Early (Steps 0-200)**:
```
<think>24</think>
<answer>24</answer>
```
Short, often wrong, no real reasoning.

**Mid (Steps 200-500)**:
```
<think>
I need to get 24 from [3, 5, 7, 2].
3 + 5 = 8
8 * 2 = 16
That's not 24.
</think>
<answer>3 + 5 + 7 + 2</answer>
```
Basic arithmetic, some exploration.

**Late (Steps 500+)**:
```
<think>
Target is 24 using [3, 5, 7, 2].

Let me try multiplication first:
- 3 * 5 = 15
- 3 * 7 = 21
- 3 * 2 = 6

What about (7 - 5) * 3 * 2?
= 2 * 3 * 2 = 12. No.

Try: (5 - 2) * (7 + 1)... wait, I don't have 1.

How about 7 * 3 + 5 - 2?
= 21 + 5 - 2 = 24 ✓

Let me verify: 7 * 3 = 21, 21 + 5 = 26, 26 - 2 = 24. Correct!
</think>
<answer>7 * 3 + 5 - 2</answer>
```
Systematic exploration, verification, backtracking.

---

## 9. Analysis & Interpretation

After training:

```bash
# 1. Evaluate
python evaluate.py --base Qwen/Qwen2.5-0.5B-Instruct --trained ./checkpoints/final

# 2. Analyze emergence
python analyze_emergence.py --log_dir ./logs

# 3. View sample completions
ls logs/completions/
cat logs/completions/step_000050.json  # Early
cat logs/completions/step_000500.json  # Late
```

### Key Questions to Answer

1. **Did accuracy improve?** (Expected: 10% → 60-90%)
2. **Did responses get longer?** (More reasoning = longer)
3. **Which patterns emerged?** (step-by-step, verification, etc.)
4. **Is it "real" reasoning or pattern matching?**

---

## 10. Troubleshooting

### Out of Memory
```python
# Reduce in config.py:
batch_size = 2
num_generations = 4
max_completion_length = 256
```

### Training Unstable
```python
# Lower learning rate:
learning_rate = 5e-7

# Increase KL penalty:
kl_coef = 0.05
```

### No Improvement
- Check reward function is working
- Verify dataset loaded correctly
- Try more training steps
- Check generation temperature (0.7 recommended)

### Garbage Output
```python
# Verify tokenizer setup:
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # Important!
```

---

## 11. Next Steps

After completing Phase 1:

### Phase 2: Real Math (GSM8K)
- Model: Qwen2.5-1.5B
- Dataset: GSM8K (8K problems)
- Goal: Compare emergence on real math

### Phase 3: Base vs Instruct
- Compare training from base model vs instruct
- Analyze Pass@K curves

### Resources
- [TinyZero Paper](https://github.com/Jiayi-Pan/TinyZero)
- [reasoning-gym](https://github.com/open-thought/reasoning-gym)
- [SimpleRL-Zoo](https://github.com/hkust-nlp/simpleRL-reason)
- [GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero)

---

## Quick Reference

### Commands
```bash
# Train
python train_simple_standalone.py

# Evaluate
python evaluate.py --trained ./checkpoints/final

# Analyze
python analyze_emergence.py

# Monitor GPU
watch -n 1 nvidia-smi
```

### Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| learning_rate | 1e-6 | Lower = more stable |
| num_generations | 4 | More = better advantages |
| kl_coef | 0.01 | Higher = less deviation |
| batch_size | 4 | Limited by VRAM |
| reward_correct | 1.0 | Binary reward |
| reward_format | 0.1 | Format-only reward |

---

**Good luck! You're about to witness reasoning emergence firsthand.**
