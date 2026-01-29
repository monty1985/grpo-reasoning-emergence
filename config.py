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