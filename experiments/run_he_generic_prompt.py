"""HumanEval generic prompt: generation + built-in detection for all 4 methods."""
import os, sys, json, math, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch, numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModelForCausalLM
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PID = os.getpid()
MODEL = "bigcode/starcoder"
N = 164
GENERIC_PROMPT = 'def solution(*args):\n    """Generate a solution\n    """\n'
# Length control: original prompt ~68 + generated ~37 = ~105 tokens
# Generic prompt 13 tokens, so need ~92 tokens generation to match
MAX_NEW_TOKENS = 95  # ≈ 105 - 13, matching original total length
HASH = 15485863
LOG_DIR = "results/generic_prompt"
os.makedirs(LOG_DIR, exist_ok=True)

print(f"[PID={PID}] HumanEval generic prompt | max_new_tokens={MAX_NEW_TOKENS}")
print(f"Prompt: {repr(GENERIC_PROMPT)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, trust_remote_code=True).to("cuda:0")
model.eval()
tcfg = TransformersConfig(model=model, tokenizer=tokenizer, vocab_size=len(tokenizer),
    device="cuda", max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.2, top_p=0.95)

# ========== Shared unwatermarked baseline ==========
print("\n=== Shared unwatermarked baseline ===")
wm_base = AutoWatermark.load("IGSW", algorithm_config="config/IGSW.json", transformers_config=tcfg)
non_texts = []
for i in tqdm(range(N), desc="non gen"):
    text = wm_base.generate_unwatermarked_text(GENERIC_PROMPT)
    if text.startswith(GENERIC_PROMPT): text = text[len(GENERIC_PROMPT):]
    non_texts.append(text)

with open(f"{LOG_DIR}/baseline.jsonl", "w") as f:
    for i in range(N):
        f.write(json.dumps({"idx": i, "unwatermarked_text": non_texts[i]}, ensure_ascii=False) + "\n")
print(f"  Saved: {LOG_DIR}/baseline.jsonl")

# ========== Watermark methods ==========
IGSW_CFG = {"algorithm_name": "IGSW", "gamma": 0.5, "delta_ref": 2.0, "hash_key": HASH,
    "z_threshold": 4.0, "prefix_length": 1, "eps": 1e-12, "visualize_mode": "raw_ig",
    "function": "tanh", "delta_min": 1.0, "delta_max": 3.0, "c0": 0.15, "k": 12}
SWEET_CFG = {"algorithm_name": "SWEET", "gamma": 0.5, "delta": 2.0, "hash_key": HASH,
    "z_threshold": 4.0, "prefix_length": 1, "entropy_threshold": 0.65}

RUNS = [
    ("IGSW", IGSW_CFG, True),
    ("KGW", "config/KGW.json", False),
    ("SWEET", SWEET_CFG, True),
    ("DBW", "config/DBW.json", False),
]

def parse_score(r):
    if isinstance(r, dict):
        for k in ["score", "z_score", "z"]:
            if k in r: return float(r[k])
    if isinstance(r, (int, float)): return float(r)
    return None

def safe_detect(wm, text):
    if not text: return None
    try:
        s = parse_score(wm.detect_watermark(text, return_dict=True))
        return s if s is not None and not math.isnan(s) else None
    except: return None

def compute_metrics(ws, ns):
    if len(ws) == 0: return {"AUROC": 0, "TPR@1%": 0, "F1@1%": 0, "TPR@5%": 0, "F1@5%": 0, "Best-F1": 0, "D": 0, "pairs": 0}
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
    D = round(float((auroc+tpr5)/2), 4)
    return {"AUROC": round(auroc, 4), "TPR@1%": round(tpr1, 4), "F1@1%": round(f1_1, 4),
            "TPR@5%": round(tpr5, 4), "F1@5%": round(f1_5, 4), "Best-F1": round(best_f1, 4),
            "D": D, "pairs": len(ws), "wm_z": round(float(np.mean(ws)), 2), "non_z": round(float(np.mean(ns)), 2)}

all_results = []
for alg, cfg, is_dict in RUNS:
    print(f"\n{'='*50}\n{alg}\n{'='*50}")
    if is_dict:
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(cfg, tmp); tmp.close()
        wm = AutoWatermark.load(alg, algorithm_config=tmp.name, transformers_config=tcfg)
        os.unlink(tmp.name)
    else:
        wm = AutoWatermark.load(alg, algorithm_config=cfg, transformers_config=tcfg)

    # Generate with generic prompt
    wm_texts = []
    for i in tqdm(range(N), desc=f"{alg} gen"):
        text = wm.generate_watermarked_text(GENERIC_PROMPT)
        if text.startswith(GENERIC_PROMPT): text = text[len(GENERIC_PROMPT):]
        wm_texts.append(text)

    jl = f"{LOG_DIR}/{alg.lower()}.jsonl"
    with open(jl, "w") as f:
        for i in range(N):
            f.write(json.dumps({"idx": i, "watermarked_text": wm_texts[i]}, ensure_ascii=False) + "\n")

    # Built-in detection
    wm_z = [s for t in wm_texts if (s := safe_detect(wm, t)) is not None]
    non_z = [s for t in non_texts if (s := safe_detect(wm, t)) is not None]
    n = min(len(wm_z), len(non_z)); m = compute_metrics(wm_z[:n], non_z[:n])
    print(f"  D={m['D']:.4f} AUROC={m['AUROC']:.4f} TPR5={m['TPR@5%']:.4f} wm_z={m['wm_z']:.2f}")
    all_results.append({"algorithm": alg, **m, "jsonl": jl})

# Summary table
print(f"\n{'='*80}")
print(f"{'Algorithm':<10} {'AUROC':>7} {'TPR1':>7} {'F1@1':>7} {'TPR5':>7} {'F1@5':>7} {'BF1':>7} {'D':>7} {'wm_z':>7}")
print("-"*80)
for r in all_results:
    print(f"{r['algorithm']:<10} {r['AUROC']:>7.4f} {r['TPR@1%']:>7.4f} {r['F1@1%']:>7.4f} {r['TPR@5%']:>7.4f} {r['F1@5%']:>7.4f} {r['Best-F1']:>7.4f} {r['D']:>7.4f} {r['wm_z']:>7.2f}")

summary = {"dataset": "HumanEval", "generic_prompt": GENERIC_PROMPT, "N": N, "results": all_results}
with open(f"{LOG_DIR}/summary.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {LOG_DIR}/summary.json\nDone!")
