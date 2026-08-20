"""Tanh c0×k 热力图 — HumanEval + StarCoder

4 c0 × 4 k = 16 组，50 随机样本
输出: results/humaneval/ablation/
"""

import os, sys, json, math, copy, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModelForCausalLM

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.dataset import HumanEvalDataset
from evaluation.tools.text_editor import TruncateTaskTextEditor, CodeGenerationTextEditor

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PID = os.getpid()
LOG_DIR = "results/humaneval/ablation"
LOG_PATH = f"{LOG_DIR}/run_tanh_{PID}.log"

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_f = open(LOG_PATH, "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_f)
sys.stderr = Tee(sys.stderr, log_f)
print(f"[PID={PID}] Tanh Heatmap | HumanEval")

MODEL_NAME = "bigcode/starcoder"
MAX_SAMPLES = 164
SUBSET = 50
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2
TOP_P = 0.95
DELTA_MIN, DELTA_MAX = 1.0, 3.5

random.seed(42)
subset_indices = sorted(random.sample(range(MAX_SAMPLES), SUBSET))

C0_GRID = [0.10, 0.15, 0.20, 0.30]
K_GRID = [4, 8, 12, 16]
n_total = len(C0_GRID) * len(K_GRID)
print(f"c0 grid: {C0_GRID}")
print(f"k  grid: {K_GRID}")
print(f"Total: {n_total} experiments")

BASE_CONFIG = {
    "algorithm_name": "IGSW", "gamma": 0.5, "delta_ref": 2.0,
    "hash_key": 15485863, "z_threshold": 4.0, "prefix_length": 1,
    "eps": 1e-12, "visualize_mode": "raw_ig",
    "delta_min": DELTA_MIN, "delta_max": DELTA_MAX,
    "function": "tanh", "k": 8, "c0": 0.20,
}

def parse_score(result):
    if isinstance(result, dict):
        for k in ["score", "z_score", "z"]:
            if k in result: return float(result[k])
        for v in result.values():
            try: return parse_score(v)
            except: pass
    if isinstance(result, (int, float)): return float(result)
    raise ValueError(f"Cannot parse: {result}")

def safe_detect(wm, text):
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return False, None
    try:
        return True, parse_score(wm.detect_watermark(text, return_dict=True))
    except:
        return False, None

def compute_metrics(wm_scores, non_scores):
    if len(wm_scores) == 0:
        return {"AUROC": 0, "TPR@1%": 0, "F1@1%": 0, "TPR@5%": 0, "F1@5%": 0,
                "Best-F1": 0, "D": 0, "z_mean": 0, "pairs": 0}
    from sklearn.metrics import roc_auc_score, roc_curve
    yt = np.array([1]*len(wm_scores) + [0]*len(non_scores))
    ys = np.array(wm_scores + non_scores)
    auroc = float(roc_auc_score(yt, ys))
    fpr, tpr, ths = roc_curve(yt, ys)

    def metrics_at(target):
        valid = np.where(fpr <= target)[0]
        if len(valid) == 0: return 0.0, 0.0
        idx = valid[np.argmax(tpr[valid])]
        tp = int(np.sum(np.array(wm_scores) >= ths[idx]))
        fp = int(np.sum(np.array(non_scores) >= ths[idx]))
        fn = len(wm_scores) - tp
        prec = tp / max(tp+fp, 1); rec = tp / max(tp+fn, 1)
        return float(rec), float(2*prec*rec/max(prec+rec, 1e-12))

    tpr1, f1_1 = metrics_at(0.01)
    tpr5, f1_5 = metrics_at(0.05)
    best_f1 = 0.0
    for th in sorted(set(ys), reverse=True):
        tp = int(np.sum(np.array(wm_scores) >= th))
        fp = int(np.sum(np.array(non_scores) >= th))
        fn = len(wm_scores) - tp
        prec = tp / max(tp+fp, 1); rec = tp / max(tp+fn, 1)
        f1 = 2*prec*rec / max(prec+rec, 1e-12)
        if f1 > best_f1: best_f1 = f1
    return {"AUROC": round(auroc, 4), "TPR@1%": round(tpr1, 4), "F1@1%": round(f1_1, 4),
            "TPR@5%": round(tpr5, 4), "F1@5%": round(f1_5, 4),
            "Best-F1": round(best_f1, 4),
            "D": round(float((auroc+tpr5)/2), 4),
            "z_mean": round(float(np.mean(wm_scores)), 4),
            "pairs": len(wm_scores)}

print("Loading StarCoder...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="cuda:0", torch_dtype=torch.float16, trust_remote_code=True)
model.eval()

tcfg = TransformersConfig(model=model, tokenizer=tokenizer, vocab_size=len(tokenizer),
    device="cuda", max_new_tokens=MAX_NEW_TOKENS, do_sample=True, top_p=TOP_P, temperature=TEMPERATURE)
dataset = HumanEvalDataset("dataset/human_eval/test.jsonl", max_samples=MAX_SAMPLES)
trunc_editor = TruncateTaskTextEditor()
code_editor = CodeGenerationTextEditor()

# 共享 unwatermarked
print("Shared unwatermarked baseline...")
os.makedirs("results/tmp_heatmap", exist_ok=True)
with open("results/tmp_heatmap/tanh_base.json", "w") as f:
    json.dump(BASE_CONFIG, f)
wm_base = AutoWatermark.load("IGSW", algorithm_config="results/tmp_heatmap/tanh_base.json", transformers_config=tcfg)

non_texts = []
for i in tqdm(subset_indices, desc="unwatermarked"):
    prompt = dataset.get_prompt(i)
    non_texts.append(code_editor.edit(
        trunc_editor.edit(wm_base.generate_unwatermarked_text(prompt), prompt), prompt))

print(f"Baseline ready: {len(non_texts)} texts")

# 跑热力图
results = []
for c0 in C0_GRID:
    for k in K_GRID:
        label = f"tanh_c0={c0}_k={k}"
        print(f"\n[{label}]")
        tmp = "results/tmp_heatmap/tanh_current.json"
        cfg = copy.deepcopy(BASE_CONFIG)
        cfg["c0"] = c0; cfg["k"] = k
        with open(tmp, "w") as f:
            json.dump(cfg, f)

        wm = AutoWatermark.load("IGSW", algorithm_config=tmp, transformers_config=tcfg)
        wm_texts = []
        for i in tqdm(range(len(non_texts)), desc=f"gen {label}", leave=False):
            prompt = dataset.get_prompt(i)
            wm_texts.append(code_editor.edit(
                trunc_editor.edit(wm.generate_watermarked_text(prompt), prompt), prompt))

        wm_scores, non_scores, skip = [], [], 0
        for i in tqdm(range(len(non_texts)), desc=f"det {label}", leave=False):
            ok_wm, sw = safe_detect(wm, wm_texts[i])
            ok_non, sn = safe_detect(wm, non_texts[i])
            if ok_wm and ok_non:
                wm_scores.append(sw); non_scores.append(sn)
            else: skip += 1

        m = compute_metrics(wm_scores, non_scores)
        r = {"c0": c0, "k": k, **m, "skipped": skip}
        print(f"  AUROC={m['AUROC']}  TPR@5%={m['TPR@5%']}  D={m['D']}  ẑ={m['z_mean']}")
        results.append(r)

# 热力图矩阵
print("\n" + "=" * 70)
print("Tanh c0×k HEATMAP")
print("=" * 70)
for metric in ["D", "AUROC", "TPR@5%"]:
    print(f"\n--- {metric} ---")
    header = "c0\\k " + "".join(f"{k:>8}" for k in K_GRID)
    print(header)
    for c0 in C0_GRID:
        vals = [next(r[metric] for r in results if r["c0"]==c0 and r["k"]==k) for k in K_GRID]
        print(f"{c0:<5}" + "".join(f"{v:>8.4f}" for v in vals))

best = max(results, key=lambda r: r["D"])
print(f"\nBest: c0={best['c0']}, k={best['k']}, D={best['D']}, AUROC={best['AUROC']}")

# 保存
with open(f"{LOG_DIR}/tanh_heatmap.json", "w") as f:
    json.dump({"c0_grid": C0_GRID, "k_grid": K_GRID, "results": results, "best": best},
              f, ensure_ascii=False, indent=2)
print(f"\nSaved: {LOG_DIR}/tanh_heatmap.json")
print("Done.")
