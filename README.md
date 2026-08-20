# IGSW + IGWD: Information Gain Guided Watermarking for Low-Entropy Code Generation

*Anonymous code supplement for conference submission.*

---

## Overview

This repository contains the implementation of **IGSW** (Information Gain guided Soft Watermarking) and **IGWD** (Information Gain Weighted Detection), a watermarking method designed for low-entropy code generation.

**Core idea:** Instead of applying a fixed-strength watermark bias at every token position, we use the *information gain* (IG) of the green-list bias — the KL divergence between the next-token distribution before and after the bias is applied — to guide both:

- **Generation (IGSW):** high IG → large bias (signal concentrated where the model is undecided and a bias can move its choice); low IG → small bias (quality preserved on committed syntax tokens)
- **Detection (IGWD):** high IG → high weight in the detection statistic, with a per-position green-list mass in the null hypothesis

Generation and detection are thus driven by a single quantity on the same tokens, forming a closed loop that is especially effective in low-entropy regimes such as code.

## Repository Structure

```
.
├── watermark/
│   ├── igsw/igsw.py          # IGSW generator (IG-guided dynamic bias)
│   ├── igwd/igwd.py          # IGWD detector (IG-weighted z-test with per-position null)
│   ├── auto_watermark.py     # Unified loader interface
│   └── auto_config.py        # Config loading
├── config/
│   ├── IGSW.json             # IGSW configuration (default paper parameters)
│   └── IGWD.json             # IGWD configuration
├── experiments/              # Reproducibility scripts for all experiments
│   ├── compute_theory.py     # Theoretical analysis (α/β/Λ)
│   ├── run_humaneval_compare.py  # Main code generation + detection
│   ├── run_mbpp_compare.py
│   ├── run_igwd_functions.py # Weight function ablation
│   ├── run_robustness.py     # Attack robustness
│   ├── run_code_attack.py    # Variable-rename attack on code
│   ├── ...
│   └── attacks/variable_rename.py
├── evaluation/               # Dataset loaders, evaluation pipelines, metrics
├── utils/                    # Model configuration, utilities
├── results/analysis/         # Summary JSONs and figures (no raw watermarked text)
└── results/figures/          # Paper figures
```

## Installation

```bash
# Python 3.10+
pip install torch transformers accelerate
pip install numpy scipy scikit-learn matplotlib
pip install libcst   # for variable-rename attack
```

## Quick Start

```python
import torch
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model (e.g., bigcode/starcoder for code, facebook/opt-1.3b for NL)
model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder").to(device)
tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder")

transformers_config = TransformersConfig(
    model=model, tokenizer=tokenizer,
    vocab_size=49152, device=device,
    max_new_tokens=200, do_sample=True,
)

# --- Generation with IGSW ---
igsw = AutoWatermark.load('IGSW', algorithm_config='config/IGSW.json',
                           transformers_config=transformers_config)
watermarked_text = igsw.generate_watermarked_text("def fibonacci(n):")
detect_result = igsw.detect_watermark(watermarked_text)
print(detect_result)

# --- Detection with IGWD ---
igwd = AutoWatermark.load('IGWD', algorithm_config='config/IGWD.json',
                           transformers_config=transformers_config)
detect_result = igwd.detect_watermark(watermarked_text)
print(detect_result)
```

## Method Summary

| Component | Description |
|---|---|
| Information Gain | $IG_t = q_G\delta_{\text{ref}} - \log Z_t$, where $q_G = g_t e^{\delta_{\text{ref}}} / Z_t$, $Z_t = 1 + (e^{\delta_{\text{ref}}}-1)g_t$ |
| IGSW bias | $\delta_t = \delta_{\min} + (\delta_{\max}-\delta_{\min}) \cdot \frac{\tanh(k(IG_t-c_0))+1}{2}$ |
| IGWD weight | $w_t = \max(IG_t - \min_t IG_t,\; 0)$ |
| IGWD statistic | $z = \frac{\sum_t w_t(x_t - g_t)}{\sqrt{\sum_t w_t^2 g_t(1-g_t)}}$ |

**Default parameters:** $\delta_{\min}=1.0$, $\delta_{\max}=3.0$, $k=12$, $c_0=0.15$, $\gamma=0.5$, hash key `15485863`, prefix length 1.

## Reproducing Experiments

All experiment scripts are in `experiments/`. Run from the project root:

```bash
# Generation + built-in detection on HumanEval
CUDA_VISIBLE_DEVICES=0 python experiments/run_humaneval_compare.py

# IGWD weight function ablation
CUDA_VISIBLE_DEVICES=0 python experiments/run_igwd_functions.py

# Theoretical analysis (fixed KGW generation, 4 detectors)
CUDA_VISIBLE_DEVICES=0 python experiments/compute_theory.py \
  --max_samples 164 --wm_file <path_to_kgw_watermarked.jsonl>

# Variable-rename attack on code
python experiments/run_code_attack.py --phase apply --attack rename --dataset HumanEval
CUDA_VISIBLE_DEVICES=0 python experiments/run_code_attack.py --phase detect --dataset HumanEval
```

## Results Summary

Selected detection quality ($D = (\text{AUROC} + \text{TPR}@5\%)/2$) on code benchmarks (built-in detector):

| Dataset | IGSW (k=12) |
|---|---|
| HumanEval (Python) | 0.872 |
| MBPP (Python) | 0.880 |
| HumanEvalPack C++ | 1.000 |
| HumanEvalPack Java | 0.996 |

IGWD as a drop-in detector further improves results; see `results/analysis/` for summary JSONs and `results/figures/` for figures.

## Notes

- This code supplement includes implementations of IGSW and IGWD along with baseline methods (KGW, SWEET, DBW, EWD) for comparison.
- Raw watermarked text JSONLs and large attack outputs are omitted from this repository for size reasons.
- For the full MarkLLM toolkit (a separate project with many additional watermarking methods), see the public MarkLLM repository.
