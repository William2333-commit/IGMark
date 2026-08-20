"""HumanEvalPack generation: C++/Java, n=5 per problem.

Usage:
  CUDA_VISIBLE_DEVICES=X python experiments/run_hep_gen.py cpp  # for C++
  CUDA_VISIBLE_DEVICES=X python experiments/run_hep_gen.py java # for Java
"""
import os, sys, json, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModelForCausalLM

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.hep_dataset import HEPDataset

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

parser = argparse.ArgumentParser()
parser.add_argument("lang", choices=["cpp", "java"])
parser.add_argument("--max-samples", type=int, default=200)
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-7B",
                    help="Model name or path (default: Qwen/Qwen2.5-Coder-7B)")
parser.add_argument("--output-dir", type=str, default=None,
                    help="Output directory (default: results/humanevalpack_qwen/{lang})")
args = parser.parse_args()

LANG = args.lang
MAX_SAMPLES = args.max_samples
MODEL = args.model
PID = os.getpid()

# Default output dir includes model hint
if args.output_dir:
    LOG_DIR = args.output_dir
else:
    model_short = MODEL.rstrip("/").split("/")[-1]
    LOG_DIR = f"results/humanevalpack_{model_short}/{LANG}"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = f"{LOG_DIR}/run_gen_{PID}.log"

class Tee:
    def __init__(self, *fs):
        self.fs = fs
    def write(self, o):
        for f in self.fs: f.write(o); f.flush()
    def flush(self):
        for f in self.fs: f.flush()

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2; TOP_P = 0.95
N_PER_PROBLEM = 1

log_f = open(LOG_PATH, "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_f)
sys.stderr = Tee(sys.stderr, log_f)
print(f"[PID={PID}] HumanEvalPack {LANG} generation  |  max_samples={MAX_SAMPLES}  n={N_PER_PROBLEM}")

def parse_score(r):
    if isinstance(r, dict):
        for k in ["score", "z_score", "z"]:
            if k in r: return float(r[k])
        for v in r.values():
            try: return parse_score(v)
            except: pass
    if isinstance(r, (int, float)): return float(r)
    raise ValueError(f"Cannot parse: {r}")

def safe_detect(wm, text):
    if not text or len(text.strip()) == 0: return None
    try:
        s = parse_score(wm.detect_watermark(text, return_dict=True))
        return s if not math.isnan(s) and not math.isinf(s) else None
    except:
        return None

def compute_metrics(ws, ns):
    if len(ws) == 0:
        return {"AUROC": 0, "TPR@1%": 0, "F1@1%": 0, "TPR@5%": 0, "F1@5%": 0, "Best-F1": 0, "D": 0, "pairs": 0}
    yt = np.array([1]*len(ws) + [0]*len(ns)); ys = np.array(ws + ns)
    auroc = float(roc_auc_score(yt, ys))
    fpr, tpr, ths = roc_curve(yt, ys)
    def m_at(t):
        valid = np.where(fpr <= t)[0]
        if len(valid) == 0: return 0.0, 0.0
        idx = valid[np.argmax(tpr[valid])]
        tp = int(np.sum(np.array(ws) >= ths[idx])); fp = int(np.sum(np.array(ns) >= ths[idx]))
        fn = len(ws) - tp
        prec = tp / max(tp+fp, 1); rec = tp / max(tp+fn, 1)
        return float(rec), float(2*prec*rec/max(prec+rec, 1e-12))
    tpr1, f1_1 = m_at(0.01); tpr5, f1_5 = m_at(0.05)
    best_f1 = 0.0
    for th in sorted(set(ys), reverse=True):
        tp = int(np.sum(np.array(ws) >= th)); fp = int(np.sum(np.array(ns) >= th))
        fn = len(ws) - tp
        prec = tp / max(tp+fp, 1); rec = tp / max(tp+fn, 1)
        f1 = 2*prec*rec / max(prec+rec, 1e-12)
        if f1 > best_f1: best_f1 = f1
    return {"AUROC": round(auroc, 4), "TPR@1%": round(tpr1, 4), "F1@1%": round(f1_1, 4),
            "TPR@5%": round(tpr5, 4), "F1@5%": round(f1_5, 4), "Best-F1": round(best_f1, 4),
            "D": round(float((auroc+tpr5)/2), 4), "pairs": len(ws)}

# ============================================================
print(f"Loading model: {MODEL}")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, trust_remote_code=True
).to("cuda:0")
model.eval()
tcfg = TransformersConfig(model=model, tokenizer=tokenizer, vocab_size=len(tokenizer),
    device="cuda", max_new_tokens=MAX_NEW_TOKENS, do_sample=True, top_p=TOP_P, temperature=TEMPERATURE)

ds = HEPDataset(language=LANG, max_samples=MAX_SAMPLES)
N = ds.prompt_nums
print(f"{LANG}: {N} problems × {N_PER_PROBLEM} = {N * N_PER_PROBLEM} outputs")

# ============================================================
# Step 1: Shared unwatermarked baseline (n=5 per problem)
# ============================================================
print("\n=== Shared unwatermarked baseline ===")
wm_base = AutoWatermark.load("IGSW", algorithm_config="config/IGSW.json", transformers_config=tcfg)

non_lists = []  # list of lists: [problem_idx][sample_idx]
for i in tqdm(range(N), desc="unwatermarked"):
    prompt = ds.get_prompt(i)
    samples = []
    for _ in range(N_PER_PROBLEM):
        text = wm_base.generate_unwatermarked_text(prompt)
        # Truncate prompt from text (keep only generated part)
        if text.startswith(prompt):
            text = text[len(prompt):]
        samples.append(text)
    non_lists.append(samples)

# Flatten for detection
non_texts_flat = [s for samples in non_lists for s in samples]
# Save baseline
baseline_data = {"non_lists": non_lists, "prompts": [ds.get_prompt(i) for i in range(N)]}
with open(f"{LOG_DIR}/baseline.jsonl", "w", encoding="utf-8") as f:
    for i, row in enumerate(non_lists):
        f.write(json.dumps({"idx": i, "prompt": ds.get_prompt(i),
                            "unwatermarked_texts": row}, ensure_ascii=False) + "\n")
print(f"  Saved: {LOG_DIR}/baseline.jsonl")

# ============================================================
# Step 2: Run each watermark method
# ============================================================
IGSW_K12_CFG = {"algorithm_name": "IGSW", "gamma": 0.5, "delta_ref": 2.0,
    "hash_key": 15485863, "z_threshold": 4.0, "prefix_length": 1,
    "eps": 1e-12, "visualize_mode": "raw_ig",
    "function": "tanh", "delta_min": 1.0, "delta_max": 3.5, "c0": 0.15, "k": 12}

os.makedirs("results/tmp_run", exist_ok=True)
SWEET_065_CFG = {"algorithm_name": "SWEET", "gamma": 0.5, "delta": 2.0,
    "hash_key": 15485863, "z_threshold": 4.0, "prefix_length": 1, "entropy_threshold": 0.65}
RUNS = [
    ("IGSW", "IGSW k=12", IGSW_K12_CFG),
    ("KGW", "KGW", "config/KGW.json"),
    ("SWEET", "SWEET", SWEET_065_CFG),
    ("DBW", "DBW", "config/DBW.json"),
]

all_results = []

for alg_name, tag, cfg in RUNS:
    print(f"\n{'='*60}\n{tag}\n{'='*60}")
    if isinstance(cfg, dict):
        tmp = f"results/tmp_run/{alg_name}_hep_{LANG}.json"
        with open(tmp, "w") as f:
            json.dump(cfg, f)
        wm = AutoWatermark.load(alg_name, algorithm_config=tmp, transformers_config=tcfg)
    else:
        wm = AutoWatermark.load(alg_name, algorithm_config=cfg, transformers_config=tcfg)

    # Generate n=5 per problem
    wm_lists = []
    for i in tqdm(range(N), desc=f"{tag} gen"):
        prompt = ds.get_prompt(i)
        samples = []
        for _ in range(N_PER_PROBLEM):
            text = wm.generate_watermarked_text(prompt)
            if text.startswith(prompt):
                text = text[len(prompt):]
            samples.append(text)
        wm_lists.append(samples)

    # Save JSONL
    jl_path = f"{LOG_DIR}/{tag.lower().replace(' ','_')}.jsonl"
    with open(jl_path, "w", encoding="utf-8") as f:
        for i in range(N):
            f.write(json.dumps({"idx": i, "prompt": ds.get_prompt(i),
                                "watermarked_texts": wm_lists[i]}, ensure_ascii=False) + "\n")
    print(f"  Saved: {jl_path}")

    # Built-in detection
    wm_texts_flat = [s for samples in wm_lists for s in samples]
    wm_scores, non_scores = [], []
    for t in tqdm(wm_texts_flat, desc=f"{tag} detect wm"):
        s = safe_detect(wm, t)
        if s is not None: wm_scores.append(s)
    for t in tqdm(non_texts_flat, desc=f"{tag} detect non"):
        s = safe_detect(wm, t)
        if s is not None: non_scores.append(s)

    nu = min(len(wm_scores), len(non_scores))
    m = compute_metrics(wm_scores[:nu], non_scores[:nu])
    D = m["D"]
    print(f"  AUROC={m['AUROC']} TPR1={m['TPR@1%']} TPR5={m['TPR@5%']} D={D} pairs={m['pairs']}")
    all_results.append({"algorithm": tag, **m, "jsonl": jl_path})

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*90}")
print(f"{'Algorithm':<14} {'AUROC':>7} {'TPR1':>7} {'F1@1':>7} {'TPR5':>7} {'F1@5':>7} {'BF1':>7} {'D':>7}")
print("-"*90)
for r in all_results:
    print(f"{r['algorithm']:<14} {r['AUROC']:>7.4f} {r['TPR@1%']:>7.4f} {r['F1@1%']:>7.4f} {r['TPR@5%']:>7.4f} {r['F1@5%']:>7.4f} {r['Best-F1']:>7.4f} {r['D']:>7.4f}")

summary = {"dataset": f"HumanEvalPack/{LANG}", "model": MODEL, "problems": N,
           "n_per_problem": N_PER_PROBLEM, "total_samples": N * N_PER_PROBLEM,
           "results": all_results}
with open(f"{LOG_DIR}/summary_gen.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {LOG_DIR}/summary_gen.json\nDone.")
