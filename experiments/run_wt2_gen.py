"""WikiText-2 generation with two decoding strategies.

Strategy 1: Sampling (temperature=0.7)
Strategy 2: Beam search (num_beams=8, no_repeat_ngram_size=16)

Usage:
  CUDA_VISIBLE_DEVICES=X python experiments/run_wt2_gen.py sample --max-samples 500
  CUDA_VISIBLE_DEVICES=X python experiments/run_wt2_gen.py beam   --max-samples 500
"""
import os, sys, json, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch, numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

parser = argparse.ArgumentParser()
parser.add_argument("strategy", choices=["sample", "beam"])
parser.add_argument("--max-samples", type=int, default=500)
args = parser.parse_args()

MAX_SAMPLES = args.max_samples
STRATEGY = args.strategy
PID = os.getpid()

if STRATEGY == "sample":
    DECODE_NAME = "sample_t0.7"
    DECODE_KWARGS = dict(do_sample=True, temperature=0.7, top_p=0.95)
else:
    DECODE_NAME = "beam8_ngram16"
    DECODE_KWARGS = dict(num_beams=8, no_repeat_ngram_size=16, early_stopping=True)

LOG_DIR = f"results/wt2/{DECODE_NAME}"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = f"{LOG_DIR}/run_{PID}.log"

class Tee:
    def __init__(self, *fs):
        self.fs = fs
    def write(self, o):
        for f in self.fs: f.write(o); f.flush()
    def flush(self):
        for f in self.fs: f.flush()

log_f = open(LOG_PATH, "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_f)
sys.stderr = Tee(sys.stderr, log_f)
print(f"[PID={PID}] WikiText-2 | {DECODE_NAME} | max_samples={MAX_SAMPLES}")

MODEL = "facebook/opt-1.3b"; MAX_NEW_TOKENS = 200; PROMPT_LEN = 50

# Load WT2, filter non-empty, build prompt/completion pairs
wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
texts = [x["text"].strip() for x in wt if x["text"].strip()]
texts = texts[:MAX_SAMPLES]
prompts, references = [], []
for t in texts:
    tokens = t.split()
    if len(tokens) < PROMPT_LEN + 20: continue  # need enough tokens
    p = " ".join(tokens[:PROMPT_LEN])
    ref = " ".join(tokens[PROMPT_LEN:PROMPT_LEN+MAX_NEW_TOKENS])
    prompts.append(p); references.append(ref)
N = len(prompts)
print(f"WT2: {N} samples (prompt={PROMPT_LEN} words, max_new={MAX_NEW_TOKENS})")

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
    except: return None

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

@torch.no_grad()
def compute_ppl(texts):
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(texts), 4):
        batch = texts[i:i+4]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda:0")
        labels = enc["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        out = model(**enc, labels=labels)
        n_tok = (labels != -100).sum().item()
        total_loss += out.loss.item() * n_tok; total_tokens += n_tok
    return round(math.exp(total_loss / max(total_tokens, 1)), 2)

print(f"Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda:0")
model.eval()
tcfg = TransformersConfig(model=model, tokenizer=tokenizer, vocab_size=len(tokenizer),
    device="cuda", max_new_tokens=MAX_NEW_TOKENS, **DECODE_KWARGS)

# Shared baseline
print("\n=== Shared unwatermarked baseline ===")
wm_base = AutoWatermark.load("IGSW", algorithm_config="config/IGSW.json", transformers_config=tcfg)
non_texts = []
for i in tqdm(range(N), desc="unwatermarked"):
    p = prompts[i]; t = wm_base.generate_unwatermarked_text(p)
    t = t[len(p):] if t.startswith(p) else t; non_texts.append(t)
non_ppl = compute_ppl(non_texts)
with open(f"{LOG_DIR}/baseline.jsonl", "w") as f:
    for i in range(N):
        f.write(json.dumps({"idx": i, "prompt": prompts[i], "reference": references[i],
                            "unwatermarked_text": non_texts[i]}, ensure_ascii=False) + "\n")
print(f"  PPL(non)={non_ppl}  Saved: {LOG_DIR}/baseline.jsonl")

# Watermark methods
IGSW_CFG = {"algorithm_name": "IGSW", "gamma": 0.5, "delta_ref": 2.0, "hash_key": 15485863,
    "z_threshold": 4.0, "prefix_length": 1, "eps": 1e-12, "visualize_mode": "raw_ig",
    "function": "tanh", "delta_min": 1.0, "delta_max": 3.0, "c0": 0.15, "k": 12}
SWEET_065 = {"algorithm_name": "SWEET", "gamma": 0.5, "delta": 2.0, "hash_key": 15485863,
    "z_threshold": 4.0, "prefix_length": 1, "entropy_threshold": 0.65}

os.makedirs("results/tmp_run", exist_ok=True)
RUNS = [
    ("IGSW", "IGSW k=12", IGSW_CFG),
    ("KGW", "KGW", "config/KGW.json"),
    ("SWEET", "SWEET", SWEET_065),
    ("DBW", "DBW", "config/DBW.json"),
]
all_results = []

for alg_name, tag, cfg in RUNS:
    print(f"\n{'='*50}\n{tag}\n{'='*50}")
    if isinstance(cfg, dict):
        tmp = f"results/tmp_run/{alg_name}_wt2.json"
        with open(tmp, "w") as f: json.dump(cfg, f)
        wm = AutoWatermark.load(alg_name, algorithm_config=tmp, transformers_config=tcfg)
    else:
        wm = AutoWatermark.load(alg_name, algorithm_config=cfg, transformers_config=tcfg)

    wm_texts = []
    for i in tqdm(range(N), desc=f"{tag} gen"):
        p = prompts[i]; t = wm.generate_watermarked_text(p)
        t = t[len(p):] if t.startswith(p) else t; wm_texts.append(t)

    jl = f"{LOG_DIR}/{tag.lower().replace(' ','_')}.jsonl"
    with open(jl, "w") as f:
        for i in range(N):
            f.write(json.dumps({"idx": i, "prompt": prompts[i], "watermarked_text": wm_texts[i]}, ensure_ascii=False) + "\n")

    ws, ns = [], []
    for t in wm_texts:
        s = safe_detect(wm, t)
        if s is not None: ws.append(s)
    for t in non_texts:
        s = safe_detect(wm, t)
        if s is not None: ns.append(s)
    nu = min(len(ws), len(ns)); m = compute_metrics(ws[:nu], ns[:nu])
    ppl = compute_ppl(wm_texts)
    print(f"  AUROC={m['AUROC']} TPR5={m['TPR@5%']} D={m['D']} PPL={ppl} pairs={m['pairs']}")
    all_results.append({"algorithm": tag, "PPL": ppl, **m, "jsonl": jl})

print(f"\n{'='*80}")
print(f"{'Algorithm':<14} {'AUROC':>7} {'TPR1':>7} {'F1@1':>7} {'TPR5':>7} {'F1@5':>7} {'BF1':>7} {'D':>7} {'PPL':>8}")
print("-"*80)
print(f"{'(unwatermarked)':<14} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {non_ppl:>8.2f}")
for r in all_results:
    print(f"{r['algorithm']:<14} {r['AUROC']:>7.4f} {r['TPR@1%']:>7.4f} {r['F1@1%']:>7.4f} {r['TPR@5%']:>7.4f} {r['F1@5%']:>7.4f} {r['Best-F1']:>7.4f} {r['D']:>7.4f} {r['PPL']:>8.2f}")

summary = {"dataset": "WikiText-2", "model": MODEL, "max_samples": N, "PPL_non": non_ppl, "results": all_results}
with open(f"{LOG_DIR}/summary.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {LOG_DIR}/summary.json\nDone.")
