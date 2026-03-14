# GraphQEC — Margulis-240 Fork

> **This is a fork of [Fadelis98/graphqec-paper](https://github.com/Fadelis98/graphqec-paper).**
> The original codebase is unchanged. This fork adds support for a custom Margulis-240 quantum error correction code.

## Custom Code Extension

This fork extends the original GraphQEC decoder to work with a **Margulis-240 code** (240 physical qubits, 8 logical qubits).

**Files added:**
- `graphqec/qecc/ldpc_code/margulis_code.py` — defines `MargulisCodeBlueprint` and `MargulisCode`
- `margulis_240.npz` — pre-computed code data (parity check matrices, permutation arrays, logical operators)
- `train_margulis.py` — training script for the GNN decoder
- `configs/train/margulis240.json` — training config

**To load the custom code:**
```python
from graphqec.qecc.ldpc_code.margulis_code import MargulisCode
code = MargulisCode.from_file('margulis_240.npz')
```

**To train the GNN decoder:**
```bash
# Clone and install
git clone https://github.com/Schulamorkodt/graphqec-paper.git
cd graphqec-paper
git checkout camera_ready
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -e .

# Run training
python train_margulis.py --config configs/train/margulis240.json
```

On a SLURM cluster:
```bash
#!/bin/bash
#SBATCH --job-name=margulis_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out

cd /path/to/graphqec-paper
python train_margulis.py --config configs/train/margulis240.json
```

Checkpoints are saved to `results/margulis240/` as `latest.pt` and `best.pt`.
