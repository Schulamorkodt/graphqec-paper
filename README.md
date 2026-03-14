# GraphQEC — Margulis-240 Fork

> **This is a fork of [Fadelis98/graphqec-paper](https://github.com/Fadelis98/graphqec-paper).**
> The original codebase is unchanged. This fork adds support for a custom Margulis-240 quantum error correction code.

## Custom Code Extension

This fork extends the original GraphQEC decoder to work with a **Margulis-240 code** (240 physical qubits, 8 logical qubits).

**Files added:**
- `graphqec/code/margulis_code.py` — defines `MargulisCodeBlueprint`, a drop-in replacement for `BBCodeBlueprint` / `ETHBBCode`
- `margulis_240.npz` — pre-computed code data (permutation matrices, parity check matrices, logical operators)

**To load the custom code:**
```python
from graphqec.code.margulis_code import MargulisCodeBlueprint

blueprint = MargulisCodeBlueprint.from_file('margulis_240.npz')
```

This replaces the original:
```python
from graphqec.code.bbcode import ETHBBCode
bb_code = ETHBBCode.from_profile('BB72_12_6')
```

---

# GraphQEC

Python package for neural-network decoding of stabilizer-based quantum error correction codes, as presented in **Towards fault-tolerant quantum computing with real-time universal neural decoding**. (The Title was "Efficient and Universal Neural-Network Decoder for Stabilizer-Based Quantum Error Correction" before this update.)

This github repo focus on providing the necessary data and code to reproduce the results in the paper. If you are interested in developing your project or training your own model, keep track of **Graphqec-lib**. We are working on refactoring the codebase to provide a more user-friendly interface for training and benchmarking, and it will be released soon.

## Features

- **Supported Codes:**
  - Sycamore Surface Codes
  - Color Codes
  - BB Codes
  - SHYPS Codes
  - 4D Toric Codes

- **Integrated Decoders:**
  - BPOSD
  - PyMatching
  - Concatenated Matching

- **Neural network decoders:**
  - `GraphRNNDecoderV5A` — A pure RNN version, excellent for small-scale codes.
  - `GraphLinearAttnDecoderV2A` — Linear attention version used in the paper, much easier to train for large codes.

## Install

### From pip

```bash
# Base installation
git clone git@github.com:Fadelis98/graphqec-paper.git
cd graphqec-paper

pip install --extra-index-url https://download.pytorch.org/whl/cu124 -e .

# For CUDA 11.8:
# pip install --extra-index-url https://download.pytorch.org/whl/cu118 -e .
```

### From uv

You can control the cuda version by editing the `pyproject.toml` if install with `uv`.

```bash
git clone git@github.com:Fadelis98/graphqec-paper.git
cd graphqec-paper

uv sync
```

Note: Choose the PyTorch version matching your CUDA version. See PyTorch installation guide for more options.

## Pretrained Models

Download trained weights from the **Releases page**.

Note that the sycamore surface code simulation depends on the original experiment data, please download it from 10.5281/zenodo.6804040.

## Usage

The `test_decoder.ipynb` in the root directory also contains a minimal example of how to decode a code using the neural network decoder, as well as an example of the full benchmarking workflow.

## Known Problems

### causal-conv1d package

`causal-conv1d` may not be compatible with some devices. It will cause an error when running the linear attention version of the neural network decoder:

```
RuntimeError: Please either install causal-conv1d>=1.4.0 to enable fast causal short convolution CUDA kernel or set use_fast_conv1d to False.
```

Turning off `use_fast_conv1d` will not fix this problem. Please view their repo to check compatibility.

It may also raise an error when installing with pip — try commenting it out in `pyproject.yaml` first and install it independently.

### `torch.compile` support

Some versions of `flash-linear-attention` missed the `@torch.compiler.disable` decorator before their Triton kernels. If `torch.compile` fails, check whether the decorator is applied to the `fused_recurrent_gated_delta_rule` kernel in `fla/ops/delta_rule/fused_recurrent`.

## Citation

```bibtex
@article{hu2025efficient,
  title={Towards fault-tolerant quantum computing with real-time universal neural decoding},
  author={Hu, Gengyuan and Ouyang, Wanli and Lu, Chao-Yang and Lin, Chen and Zhong, Han-Sen},
  journal={arXiv preprint arXiv:2502.19971},
  year={2025}
}
```

Note: The paper title has been updated. For the old version, see below:

```bibtex
@article{hu2025efficient,
  title={Efficient and Universal Neural-Network Decoder for Stabilizer-Based Quantum Error Correction},
  author={Hu, Gengyuan and Ouyang, Wanli and Lu, Chao-Yang and Lin, Chen and Zhong, Han-Sen},
  journal={arXiv preprint arXiv:2502.19971},
  year={2025}
}
```
