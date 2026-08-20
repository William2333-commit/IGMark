"""IGWD with 4 weight functions (Linear, Sigmoid, Exponential, Tanh)
on HumanEval + MBPP, for all 4 generators (IGSW, KGW, SWEET, DBW).
"""
import sys, json, math, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch, numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

TARGET_DS = sys.argv[1]  # HumanEval or MBPP
GAMMA=0.5; PREFIX=1; HASH=15485863; DREF=2.0; EPS=1e-12

if TARGET_DS == "HumanEval":
    MODEL = "bigcode/starcoder"
    FOLDER = "results/humaneval/compare"
    N = 164
else:
    MODEL = "bigcode/starcoder"
    FOLDER = "results/mbpp/compare"
    N = 500

PID = os.getpid()
print(f"[PID={PID}] IGWD functions: {TARGET_DS} | Linear/Sigmoid/Exp/Tanh × IGSW/KGW/SWEET/DBW")

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, trust_remote_code=True).to("cuda:0")
model.eval()

# Load texts
if TARGET_DS == "HumanEval":
    igsw_file = next(f for f in os.listdir(FOLDER) if f.startswith("igsw_k12") and f.endswith(".jsonl"))
    with open(f"{FOLDER}/{igsw_file}") as f:
        bdata = [json.loads(l) for l in f][:N]
    non_texts = [d["unwatermarked_text"] for d in bdata]
    non_prompts = [d.get("prompt", "") for d in bdata]
    gen_files = {"IGSW": "igsw_k12.jsonl", "KGW": "kgw.jsonl", "SWEET": "sweet.jsonl", "DBW": "dbw.jsonl"}
else:
    import pickle
    with open("results/mbpp/shared_baseline/baseline.pkl", "rb") as f:
        non_texts = pickle.load(f)["non_texts"][:N]
    with open(f"{FOLDER}/igsw_k=12.jsonl") as f:
        non_prompts = [json.loads(l)["prompt"] for l in f][:N]
    gen_files = {"IGSW": "igsw_k=12.jsonl", "KGW": "kgw.jsonl", "SWEET": "sweet_065.jsonl", "DBW": "dbw.jsonl"}

N = min(len(non_texts), N)
non_texts = non_texts[:N]; non_prompts = non_prompts[:N]
print(f"Non texts: {len(non_texts)}")

# Load wm texts
wm_texts_all = {}
for alg in ["IGSW", "KGW", "SWEET", "DBW"]:
    path = f"{FOLDER}/{gen_files[alg]}"
    if os.path.exists(path):
        with open(path) as f:
            data = [json.loads(l) for l in f]
        wm_texts_all[alg] = [d.get("watermarked_text", d.get("watermarked", "")) for d in data][:N]
        print(f"  {alg}: {len(wm_texts_all[alg])}")

# ============ Greenlist + IG utilities ============
def get_gl(prefix_ids, device, vocab_size):
    tr = 1
    for t in prefix_ids: tr *= int(t)
    rng = torch.Generator(device=device)
    rng.manual_seed(HASH * (tr % vocab_size))
    return torch.randperm(vocab_size, device=device, generator=rng)[:int(vocab_size * GAMMA)]

def ig_from_gm(g, d, ep=1e-12):
    g = max(0.0, min(1.0, float(g)))
    a = math.exp(float(d)); z = 1.0 + (a - 1.0) * g
    return float(max((a*g)/max(z,ep)*float(d) - math.log(max(z,ep)), 0.0))

# ============ Weight functions ============
def weight_linear(igs):
    """w = (ig - min_ig) / range, normalized"""
    ig = np.array(igs); rng = max(ig.max() - ig.min(), 1e-12)
    return np.maximum((ig - ig.min()) / rng, 0.0)

def weight_sigmoid(igs, k=12, c0=0.15):
    """w = 1 / (1 + exp(-k*(ig - c0))) — flipped: high w for high IG"""
    ig = np.array(igs)
    return 1.0 / (1.0 + np.exp(-k * (ig - c0)))

def weight_exponential(igs, k=12):
    """w = 1 - exp(-k * ig) — flipped: high w for high IG"""
    ig = np.array(igs)
    return 1.0 - np.exp(-k * np.maximum(ig, 0.0))

def weight_tanh(igs, k=12, c0=0.15):
    """w = (tanh(k*(ig-c0))+1)/2 — flipped: high w for high IG"""
    ig = np.array(igs)
    return (np.tanh(k * (ig - c0)) + 1.0) / 2.0

WEIGHT_FUNCS = {
    "Linear":      weight_linear,
    "Sigmoid":     weight_sigmoid,
    "Exponential": weight_exponential,
    "Tanh":        weight_tanh,
}

# ============ Feature computation ============
@torch.no_grad()
def compute_features(text, prompt=""):
    if not text or len(text.strip()) == 0:
        return None
    dev = next(model.parameters()).device
    pids = tokenizer.encode(prompt, add_special_tokens=False) if prompt else []
    cids = tokenizer.encode(text, add_special_tokens=False)
    if len(cids) == 0: return None
    fids = pids + cids; sp = len(pids)
    logits = model(input_ids=torch.tensor([fids[:-1]], dtype=torch.long, device=dev)).logits[0]
    gf, igs = [], []
    for pos in range(sp+PREFIX, len(fids)):
        if pos-1 >= len(logits): continue
        prefix = fids[max(0, pos-PREFIX):pos]
        probs = torch.softmax(logits[pos-1].float(), dim=-1)
        gl = get_gl(prefix, dev, len(tokenizer))
        gf.append(1.0 if (gl == fids[pos]).any().cpu().item() else 0.0)
        gm = float(torch.sum(probs[gl]).cpu())
        igs.append(ig_from_gm(gm, DREF))
    return {"green": np.array(gf), "ig": np.array(igs)}

def igwd_zscore(features, weight_fn_name):
    f = features
    if f is None or len(f["ig"]) == 0: return None
    w = WEIGHT_FUNCS[weight_fn_name](f["ig"])
    sw = float(np.sum(w)); sw2 = float(np.sum(w*w))
    if sw2 < EPS: return None
    return float((np.sum(w * f["green"]) - GAMMA * sw) / math.sqrt(GAMMA * (1.0 - GAMMA) * sw2))

# ============ Metrics ============
def compute_metrics(ws, ns):
    if len(ws) == 0:
        return {"AUROC": 0, "TPR@1%": 0, "F1@1%": 0, "TPR@5%": 0, "F1@5%": 0, "Best-F1": 0, "D": 0}
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
            "TPR@5%": round(tpr5, 4), "F1@5%": round(f1_5, 4), "Best-F1": round(best_f1, 4), "D": D}

# ============ Main ============
# Precompute non features
print("Precomputing non features...")
non_features = [compute_features(non_texts[i], non_prompts[i]) for i in tqdm(range(N), desc="non")]

all_results = []

for gen_alg in ["IGSW", "KGW", "SWEET", "DBW"]:
    if gen_alg not in wm_texts_all: continue
    wts = wm_texts_all[gen_alg]
    prompts_wm = [non_prompts[i % N] for i in range(len(wts))]
    print(f"\n{'='*50}\n{gen_alg}\n{'='*50}")

    # Precompute wm features
    wm_features = [compute_features(wts[i], prompts_wm[i]) for i in tqdm(range(len(wts)), desc=f"{gen_alg} feat")]

    for wf_name in WEIGHT_FUNCS:
        print(f"  {wf_name}...")
        wm_zs = [s for f in wm_features if (s := igwd_zscore(f, wf_name)) is not None]
        non_zs = [s for f in non_features if (s := igwd_zscore(f, wf_name)) is not None]
        n = min(len(wm_zs), len(non_zs))
        m = compute_metrics(wm_zs[:n], non_zs[:n])
        print(f"    D={m['D']:.4f} AUROC={m['AUROC']:.4f} TPR5={m['TPR@5%']:.4f}")
        all_results.append({"generator": gen_alg, "weight_func": wf_name, **m, "pairs": n})

# Summary
print(f"\n{'='*90}")
print(f"{'Gen':<6} {'Weight':<12} {'AUROC':>7} {'TPR1':>7} {'F1@1':>7} {'TPR5':>7} {'F1@5':>7} {'BF1':>7} {'D':>7}")
print("-"*90)
for r in all_results:
    print(f"{r['generator']:<6} {r['weight_func']:<12} {r['AUROC']:>7.4f} {r['TPR@1%']:>7.4f} {r['F1@1%']:>7.4f} {r['TPR@5%']:>7.4f} {r['F1@5%']:>7.4f} {r['Best-F1']:>7.4f} {r['D']:>7.4f}")

OUT_DIR = f"results/igwd_functions/{TARGET_DS}"
os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump({"dataset": TARGET_DS, "results": all_results}, f, ensure_ascii=False, indent=2)
print(f"\nSaved: results/analysis/igwd_functions_{TARGET_DS}.json\nDone.")
