"""IGSW 消融实验 v2（C4 + OPT-1.3B，DBW 风格）

实验顺序:
  Step 1: δ 范围 → 固定 [1.0, 3.5]（已由 v1 确定）
  Step 2: 水印强度函数对比
          - 每个函数扫关键参数 → 多个 (PPL, ẑ) 点
          - 二阶多项式拟合 PPL = f(ẑ)
          - 交叉验证比较整条曲线（DBW Fig 3a 方法）
  Step 3: 最优函数的 c0 和 k 消融（DBW Fig 3b）

输出: results/c4/ablation_v2/
"""

import os, sys, json, math, copy
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
from evaluation.dataset import C4Dataset
from evaluation.tools.text_editor import TruncateTaskTextEditor

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ============================================================
# 日志
# ============================================================
PID = os.getpid()
LOG_DIR = "results/c4/ablation_v2"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = f"{LOG_DIR}/run_{PID}.log"

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
print(f"[PID={PID}] Log: {LOG_PATH}")

# ============================================================
# 配置
# ============================================================
MODEL_NAME = "facebook/opt-1.3b"
DATASET_PATH = "dataset/c4/processed_c4.json"
MAX_SAMPLES = 50
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.95

# δ 范围已由 v1 确定
DELTA_MIN, DELTA_MAX = 1.0, 3.5

BASE_CONFIG = {
    "algorithm_name": "IGSW", "gamma": 0.5, "delta_ref": 2.0,
    "hash_key": 15485863, "z_threshold": 4.0, "prefix_length": 1,
    "eps": 1e-12, "visualize_mode": "raw_ig",
    "delta_min": DELTA_MIN, "delta_max": DELTA_MAX,
    "k": 8, "c0": 0.20,
    "function": "sigmoid"
}

# ============================================================
# 工具函数
# ============================================================
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
        return {"AUROC": 0, "TPR@5%": 0, "D": 0, "z_mean": 0, "pairs": 0}
    yt = np.array([1]*len(wm_scores) + [0]*len(non_scores))
    ys = np.array(wm_scores + non_scores)
    auroc = float(roc_auc_score(yt, ys))
    fpr, tpr, _ = roc_curve(yt, ys)
    valid = np.where(fpr <= 0.05)[0]
    tpr5 = float(tpr[valid[-1]]) if len(valid) > 0 else 0.0
    return {"AUROC": round(auroc, 4), "TPR@5%": round(tpr5, 4),
            "D": round(float((auroc+tpr5)/2), 4),
            "z_mean": round(float(np.mean(wm_scores)), 4),
            "pairs": len(wm_scores)}

def compute_ppl(model, tokenizer, text, prompt):
    full = prompt + text
    enc = tokenizer.encode(full, return_tensors="pt").to(model.device)
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    labels = enc.clone()
    labels[0, :prompt_len] = -100
    with torch.no_grad():
        loss = model(enc, labels=labels).loss
    return round(math.exp(loss.item()), 2)

def make_config(overrides):
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg.update(overrides)
    return cfg

def evaluate_one(model, tokenizer, overrides, label, non_texts, N, dataset,
                 trunc_editor, tcfg):
    """单次评测"""
    tmp_path = "results/tmp_ablation/current.json"
    os.makedirs("results/tmp_ablation", exist_ok=True)
    with open(tmp_path, "w") as f:
        json.dump(make_config(overrides), f)

    wm = AutoWatermark.load("IGSW", algorithm_config=tmp_path, transformers_config=tcfg)

    # 生成 + PPL
    wm_texts, ppls = [], []
    for i in tqdm(range(N), desc=f"gen {label}", leave=False):
        prompt = dataset.get_prompt(i)
        full = trunc_editor.edit(wm.generate_watermarked_text(prompt), prompt)
        wm_texts.append(full)
        ppls.append(compute_ppl(model, tokenizer, full, prompt))

    ppl_mean = round(float(np.mean(ppls)), 2)

    # 检测
    wm_scores, non_scores, skip = [], [], 0
    for i in tqdm(range(N), desc=f"det {label}", leave=False):
        ok_wm, sw = safe_detect(wm, wm_texts[i])
        ok_non, sn = safe_detect(wm, non_texts[i])
        if ok_wm and ok_non:
            wm_scores.append(sw); non_scores.append(sn)
        else:
            skip += 1

    metrics = compute_metrics(wm_scores, non_scores)
    print(f"  [{label}] PPL={ppl_mean}  ẑ={metrics['z_mean']}  AUROC={metrics['AUROC']}  skip={skip}")
    return {"ppl": ppl_mean, **overrides, **metrics, "skipped": skip}

# ============================================================
# 加载模型 & 共享基线
# ============================================================
print("=" * 60)
print(f"IGSW Ablation v2 | {MODEL_NAME} | C4 | {MAX_SAMPLES} samples")
print(f"δ range: [{DELTA_MIN}, {DELTA_MAX}]")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="cuda:0", torch_dtype=torch.float16, trust_remote_code=True)
model.eval()
model.config.pad_token_id = tokenizer.pad_token_id

tcfg = TransformersConfig(
    model=model, tokenizer=tokenizer, vocab_size=len(tokenizer),
    device="cuda", max_new_tokens=MAX_NEW_TOKENS,
    do_sample=True, top_p=TOP_P, temperature=TEMPERATURE)

dataset = C4Dataset(DATASET_PATH, max_samples=MAX_SAMPLES)
N = dataset.prompt_nums
print(f"C4: {N} samples")

# 共享 unwatermarked 基线
print("\nShared unwatermarked baseline...")
os.makedirs("results/tmp_ablation", exist_ok=True)
with open("results/tmp_ablation/base_config.json", "w") as f:
    json.dump(BASE_CONFIG, f)

wm_base = AutoWatermark.load("IGSW", algorithm_config="results/tmp_ablation/base_config.json",
                             transformers_config=tcfg)
trunc_editor = TruncateTaskTextEditor()

non_texts, non_ppls = [], []
for i in tqdm(range(N), desc="unwatermarked"):
    prompt = dataset.get_prompt(i)
    full = trunc_editor.edit(wm_base.generate_unwatermarked_text(prompt), prompt)
    non_texts.append(full)
    non_ppls.append(compute_ppl(model, tokenizer, full, prompt))

ppl_non = round(float(np.mean(non_ppls)), 2)
print(f"Unwatermarked PPL = {ppl_non}\n")

def run(overrides, label):
    return evaluate_one(model, tokenizer, overrides, label, non_texts, N,
                       dataset, trunc_editor, tcfg)

# ============================================================
# Step 2: 函数对比（扫参数 + 二阶拟合）
# ============================================================
print("=" * 70)
print("Step 2: Function comparison with parameter sweeps")
print("=" * 70)

# 每个函数的参数扫瞄定义
FUNCTION_SWEEPS = [
    {"function": "sigmoid",     "param_name": "k", "c0": 0.20,
     "params": [4, 6, 8, 10, 12, 16],
     "label_fmt": "Sig k={param}"},
    {"function": "linear",      "param_name": "k", "c0": 0.20,
     "params": [0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
     "label_fmt": "Lin w={param}"},
    {"function": "step",        "param_name": "c0", "k": 0,
     "params": [0.15, 0.20, 0.25, 0.30, 0.35],
     "label_fmt": "Step c0={param}"},
    {"function": "tanh",        "param_name": "k", "c0": 0.20,
     "params": [4, 6, 8, 10, 12, 16],
     "label_fmt": "Tanh k={param}"},
    {"function": "exponential", "param_name": "k", "c0": 0.20,
     "params": [4, 6, 8, 10, 12, 16],
     "label_fmt": "Exp k={param}"},
    {"function": "logarithmic", "param_name": "k", "c0": 0.20,
     "params": [4, 6, 8, 10, 12, 16],
     "label_fmt": "Log k={param}"},
    {"function": "piecewise",   "param_name": "k", "c0": 0.20,
     "params": [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
     "label_fmt": "PW w={param}"},
]

all_sweep_results = []
function_curves = {}  # function_name -> {"points": [(ppl,z),...], "fit": coeffs}

for fs in FUNCTION_SWEEPS:
    fn_name = fs["function"]
    print(f"\n{'='*50}")
    print(f"Function: {fn_name}  |  sweeping {fs['param_name']}")
    print(f"{'='*50}")

    points = []
    for pv in fs["params"]:
        overrides = {"function": fn_name}
        # Set the swept parameter
        if fs["param_name"] == "k":
            overrides["k"] = pv
            overrides["c0"] = fs["c0"]
        elif fs["param_name"] == "c0":
            overrides["c0"] = pv
            overrides["k"] = fs.get("k", 0)

        label = fs["label_fmt"].format(param=pv)
        r = run(overrides, label)
        points.append({"param": pv, "ppl": r["ppl"], "z_mean": r["z_mean"],
                       "AUROC": r["AUROC"], "D": r["D"]})
        all_sweep_results.append(r)

    # 二阶多项式拟合: PPL = a*ẑ² + b*ẑ + c
    zs = np.array([p["z_mean"] for p in points])
    ppls = np.array([p["ppl"] for p in points])

    if len(zs) >= 3 and len(set(zs)) >= 3:
        coeffs = np.polyfit(zs, ppls, 2)  # [a, b, c]
        fitted = np.polyval(coeffs, zs)
        r2 = 1 - np.sum((ppls - fitted)**2) / np.sum((ppls - ppls.mean())**2)
        print(f"  Fit: PPL = {coeffs[0]:.4f}·ẑ² + {coeffs[1]:.4f}·ẑ + {coeffs[2]:.4f}  R²={r2:.4f}")
    else:
        coeffs = None
        r2 = 0
        print(f"  Cannot fit (need >=3 unique ẑ values)")

    function_curves[fn_name] = {"points": points, "fit_coeffs": coeffs.tolist() if coeffs is not None else None,
                                "r2": round(float(r2), 4)}

# 比较函数：在 ẑ 的公共范围内，计算各函数拟合曲线的平均 PPL
# 公共 ẑ 范围
all_z = []
for fc in function_curves.values():
    for p in fc["points"]:
        all_z.append(p["z_mean"])
z_min, z_max = min(all_z), max(all_z)
z_grid = np.linspace(z_min, z_max, 100)

print("\n" + "=" * 70)
print("Function Comparison (by fitted curve)")
print("=" * 70)
print(f"ẑ range: [{z_min:.2f}, {z_max:.2f}]")
print(f"{'Function':<14} {'R²':>6} {'Avg fitted PPL':>14}  {'Best param PPL':>14}  {'Best param ẑ':>14}")
print("-" * 70)

best_function = None
best_avg_ppl = float('inf')
for fn_name, fc in function_curves.items():
    if fc["fit_coeffs"] is not None:
        a, b, c = fc["fit_coeffs"]
        fitted_grid = a * z_grid**2 + b * z_grid + c
        avg_ppl = float(np.mean(fitted_grid))
        best_point = min(fc["points"], key=lambda p: p["ppl"] / max(p["z_mean"], 1e-6))
        print(f"{fn_name:<14} {fc['r2']:>6.4f} {avg_ppl:>14.4f}  {best_point['ppl']:>14.2f}  {best_point['z_mean']:>14.4f}")
        if avg_ppl < best_avg_ppl:
            best_avg_ppl = avg_ppl
            best_function = fn_name
    else:
        best_point = min(fc["points"], key=lambda p: p["ppl"] / max(p["z_mean"], 1e-6))
        print(f"{fn_name:<14} {'N/A':>6} {'N/A':>14}  {best_point['ppl']:>14.2f}  {best_point['z_mean']:>14.4f}")

print(f"\n>>> Best function by fitted curve: {best_function}")

# 确定最优函数的参数
best_func_points = function_curves[best_function]["points"]
best_func_param = min(best_func_points, key=lambda p: p["ppl"] / max(p["z_mean"], 1e-6))

with open(f"{LOG_DIR}/step2_function_comparison.json", "w") as f:
    json.dump({
        "ppl_non": ppl_non, "delta_range": f"[{DELTA_MIN},{DELTA_MAX}]",
        "best_function_by_curve": best_function,
        "function_curves": {k: {
            "points": v["points"],
            "fit_coeffs": v["fit_coeffs"], "r2": v["r2"]
        } for k, v in function_curves.items()},
    }, f, ensure_ascii=False, indent=2)

# ============================================================
# Step 3: c0 和 k 消融
# ============================================================
BEST_FN = best_function
BEST_PARAM_K = best_func_param["param"]  # the best value from the sweep

print(f"\n{'='*70}")
print(f"Step 3a: c0 ablation ({BEST_FN}, k={BEST_PARAM_K})")
print("=" * 70)

C0_VALS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
step3a_results = []
for c0 in C0_VALS:
    step3a_results.append(run({"function": BEST_FN, "k": BEST_PARAM_K, "c0": c0}, f"c0={c0}"))

best_c0 = min(step3a_results, key=lambda x: x["ppl"] / max(x["z_mean"], 1e-6))
print(f"\n>>> Best c0 = {best_c0['c0']}  PPL={best_c0['ppl']}  ẑ={best_c0['z_mean']}")

print(f"\n{'='*70}")
print(f"Step 3b: k ablation ({BEST_FN}, c0={best_c0['c0']})")
print("=" * 70)

if BEST_FN in ["step"]:
    K_VALS = [0, 0, 0]  # step has no k
elif BEST_FN in ["linear", "piecewise"]:
    K_VALS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
else:
    K_VALS = [4, 6, 8, 10, 12, 16]

step3b_results = []
for k in K_VALS:
    step3b_results.append(run({"function": BEST_FN, "k": k, "c0": best_c0["c0"]}, f"k={k}"))

best_k = min(step3b_results, key=lambda x: x["ppl"] / max(x["z_mean"], 1e-6))
print(f"\n>>> Best k = {best_k['k']}  PPL={best_k['ppl']}  ẑ={best_k['z_mean']}")

with open(f"{LOG_DIR}/step3_params.json", "w") as f:
    json.dump({
        "function": BEST_FN, "ppl_non": ppl_non,
        "c0_sweep": {"k_fixed": BEST_PARAM_K, "results": step3a_results, "best_c0": best_c0["c0"]},
        "k_sweep": {"c0_fixed": best_c0["c0"], "results": step3b_results, "best_k": best_k["k"]},
    }, f, ensure_ascii=False, indent=2)

# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 90)
print("ABLATION SUMMARY")
print("=" * 90)

print(f"\n--- Function Comparison (fitted curve PPL vs ẑ) ---")
for fn_name, fc in function_curves.items():
    pts = fc["points"]
    ppls_str = ", ".join(f"{p['ppl']:.2f}" for p in pts)
    zs_str = ", ".join(f"{p['z_mean']:.2f}" for p in pts)
    fit_str = f"  fit: PPL={fc['fit_coeffs'][0]:.4f}·ẑ²+{fc['fit_coeffs'][1]:.4f}·ẑ+{fc['fit_coeffs'][2]:.4f}" if fc['fit_coeffs'] else ""
    print(f"  {fn_name:<14} PPL=[{ppls_str}]")
    print(f"  {'':14} ẑ  =[{zs_str}]")
    if fit_str:
        print(f"  {'':14}{fit_str}")

print(f"\n--- Step 3a: c0 sweep ---")
for r in step3a_results:
    print(f"  c0={r['c0']}  PPL={r['ppl']}  ẑ={r['z_mean']}")

print(f"\n--- Step 3b: k sweep ---")
for r in step3b_results:
    print(f"  k={r['k']}  PPL={r['ppl']}  ẑ={r['z_mean']}")

print(f"\n>>> 最优参数: function={BEST_FN}, c0={best_c0['c0']}, k={best_k['k']}")
print(f"    δ=[{DELTA_MIN},{DELTA_MAX}], PPL={best_k['ppl']}, ẑ={best_k['z_mean']}")

summary = {
    "dataset": "C4", "model": MODEL_NAME, "max_samples": MAX_SAMPLES,
    "delta": [DELTA_MIN, DELTA_MAX], "ppl_non": ppl_non,
    "step2_functions": {
        "best_by_curve": best_function,
        "curves": {k: {"points": v["points"], "fit_coeffs": v["fit_coeffs"], "r2": v["r2"]}
                   for k, v in function_curves.items()},
    },
    "step3_params": {
        "best_c0": best_c0["c0"], "best_k": best_k["k"],
    },
    "optimal": {
        "delta_min": DELTA_MIN, "delta_max": DELTA_MAX,
        "function": BEST_FN, "c0": best_c0["c0"], "k": best_k["k"],
        "ppl": best_k["ppl"], "z_mean": best_k["z_mean"],
    },
}
with open(f"{LOG_DIR}/ablation_summary.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\nAll saved to {LOG_DIR}/")
print("Done.")
