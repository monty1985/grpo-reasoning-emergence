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
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
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