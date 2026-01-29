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