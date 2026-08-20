"""GPU 2: HumanEval Detection v3 — fixed greenlist via watermark utils"""
import os, sys, json, math
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch, numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModelForCausalLM
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig

PID = os.getpid()
LOG_DIR = "results/humaneval/compare"
LOG_PATH = f"{LOG_DIR}/run_detection_v3_{PID}.log"

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
print(f"[GPU2 PID={PID}] HumanEval Detection v3 — fixed greenlist")

MODEL = "bigcode/starcoder"
GAMMA = 0.5; PREFIX = 1; TAU = 1.0; EPS = 1e-12; DREF = 2.0

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Shared non texts from igsw_k12
with open(f"{LOG_DIR}/igsw_k12.jsonl") as f:
    baseline = [json.loads(l) for l in f]
non_texts = [d["unwatermarked_text"] for d in baseline]
non_prompts = [d["prompt"] for d in baseline]
N = len(non_texts)
print(f"Shared non_texts: {N}")

print("Loading StarCoder...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, trust_remote_code=True).to("cuda:0")
model.eval()
tcfg = TransformersConfig(model=model, tokenizer=tokenizer, vocab_size=len(tokenizer),
    device="cuda", max_new_tokens=512, do_sample=True, top_p=0.95, temperature=0.2)

# Load all watermarks for greenlist
wms = {}
for alg in ["KGW", "SWEET", "DBW", "IGSW"]:
    wms[alg] = AutoWatermark.load(alg, algorithm_config=f"config/{alg}.json", transformers_config=tcfg)
    print(f"  Loaded {alg}")

def get_utils(w):
    for n in ["utils", "watermark_utils", "algorithm_utils"]:
        if hasattr(w, n): return getattr(w, n)

def get_gl(wm, prefix_ids, device):
    utils = get_utils(wm)
    for x in [torch.tensor(prefix_ids, dtype=torch.long, device=device),
              torch.tensor(prefix_ids, dtype=torch.long, device="cpu"), prefix_ids]:
        try:
            ids = utils.get_greenlist_ids(x)
            if isinstance(ids, tuple): ids = ids[0]
            if isinstance(ids, list): ids = torch.tensor(ids, dtype=torch.long, device=device)
            elif isinstance(ids, np.ndarray): ids = torch.tensor(ids, dtype=torch.long, device=device)
            elif torch.is_tensor(ids): ids = ids.to(device)
            else: ids = torch.tensor(list(ids), dtype=torch.long, device=device)
            return ids.view(-1)
        except: pass
    raise RuntimeError

def spike_entropy(probs, tau):
    return float(torch.sum(probs / (1.0 + tau * probs)).detach().cpu())

def ig_from_green_mass(g, d, eps_=1e-12):
    g = max(0.0, min(1.0, float(g)))
    a = math.exp(float(d)); z = 1.0 + (a - 1.0) * g
    return float(max((a*g)/max(z,eps_)*float(d) - math.log(max(z,eps_)), 0.0))

def linear_w(xs, eps_=1e-12):
    x = np.array(xs, dtype=float)
    if len(x) == 0: return x
    c0 = float(np.min(x)); w = np.maximum(x - c0, 0.0)
    return np.ones_like(x) if float(np.sum(w*w)) <= eps_ else w.astype(float)

@torch.no_grad()
def compute_features(wm, prompt, text):
    if not text or len(str(text).strip()) == 0:
        return {"green": [], "entropy": [], "ig": []}
    dev = next(model.parameters()).device
    pids = tokenizer.encode(prompt, add_special_tokens=False)
    cids = tokenizer.encode(text, add_special_tokens=False)
    if len(cids) == 0: return {"green": [], "entropy": [], "ig": []}
    fids = pids + cids
    if len(fids) < 2: return {"green": [], "entropy": [], "ig": []}
    logits = model(input_ids=torch.tensor([fids[:-1]], dtype=torch.long, device=dev)).logits[0]
    sp = len(pids)
    gf, se, igs = [], [], []
    for pos in range(sp, len(fids)):
        if pos <= 0: continue
        prefix = fids[max(0, pos - PREFIX):pos]
        if len(prefix) == 0: continue
        probs = torch.softmax(logits[pos - 1].float(), dim=-1)
        gids = get_gl(wm, prefix, probs.device)
        gf.append(1.0 if bool((gids == fids[pos]).any().cpu().item()) else 0.0)
        gm = float(torch.sum(probs[gids]).cpu())
        se.append(spike_entropy(probs, TAU))
        igs.append(ig_from_green_mass(gm, DREF, EPS))
    return {"green": gf, "entropy": se, "ig": igs}

def score_ewd(f):
    g = f["green"]; se = f["entropy"]
    if len(g) == 0: return None
    w = linear_w(se, EPS)
    denom = GAMMA * (1.0 - GAMMA) * float(np.sum(w*w))
    return float((np.sum(w*np.array(g)) - GAMMA*float(np.sum(w))) / math.sqrt(denom)) if denom > EPS else None

def score_igwd(f):
    g = f["green"]; ig = f["ig"]
    if len(g) == 0: return None
    w = linear_w(ig, EPS)
    denom = GAMMA * (1.0 - GAMMA) * float(np.sum(w*w))
    return float((np.sum(w*np.array(g)) - GAMMA*float(np.sum(w))) / math.sqrt(denom)) if denom > EPS else None

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

# ---- Precompute non features (use KGW for greenlist) ----
print("Precomputing non features...")
non_wm = wms["KGW"]  # all use same hash scheme
non_features = [compute_features(non_wm, non_prompts[i], non_texts[i]) for i in tqdm(range(N), desc="non")]

# ---- Detection pairs ----
PAIRS = [
    ("igsw_k12.jsonl", "IGSW k=12", "IGSW"),
    ("kgw.jsonl", "KGW", "KGW"),
    ("sweet.jsonl", "SWEET", "SWEET"),
    ("dbw.jsonl", "DBW", "DBW"),
]
DETECTORS = {"EWD": score_ewd, "IGWD": score_igwd}
all_results = []

for jl_file, gname, alg_key in PAIRS:
    with open(f"{LOG_DIR}/{jl_file}") as f:
        data = [json.loads(l) for l in f]
    prompts = [d["prompt"] for d in data]
    wts_texts = [d["watermarked_text"] for d in data]
    wm = wms[alg_key]
    print(f"\n{'='*50}\n{gname} ({jl_file}): {len(wts_texts)} texts  |  greenlist={alg_key}\n{'='*50}")

    for dname, sf in DETECTORS.items():
        non_scores = [s for fn in non_features if (s := sf(fn)) is not None]
        ws, sk = [], 0
        for i in tqdm(range(len(wts_texts)), desc=f"{gname}/{dname}"):
            fw = compute_features(wm, prompts[i], wts_texts[i])
            s = sf(fw)
            if s is not None: ws.append(s)
            else: sk += 1
        nu = min(len(ws), len(non_scores))
        m = compute_metrics(ws[:nu], non_scores[:nu])
        print(f"  {dname}: AUROC={m['AUROC']} TPR1={m['TPR@1%']} F1@1={m['F1@1%']} TPR5={m['TPR@5%']} F1@5={m['F1@5%']} BF1={m['Best-F1']} D={m['D']} p={m['pairs']} sk={sk}")
        all_results.append({"generation": gname, "detection": dname,
            "AUROC": m["AUROC"], "TPR@1%": m["TPR@1%"], "F1@1%": m["F1@1%"],
            "TPR@5%": m["TPR@5%"], "F1@5%": m["F1@5%"], "Best-F1": m["Best-F1"],
            "D": m["D"], "pairs": m["pairs"], "skipped": sk})

# ---- Table ----
print(f"\n{'='*90}")
print(f"{'Gen':<10} {'Det':<7} {'AUROC':>7} {'TPR1':>7} {'F1@1':>7} {'TPR5':>7} {'F1@5':>7} {'BF1':>7} {'D':>7}")
print("-"*90)
for r in all_results:
    print(f"{r['generation']:<10} {r['detection']:<7} {r['AUROC']:>7.4f} {r['TPR@1%']:>7.4f} {r['F1@1%']:>7.4f} {r['TPR@5%']:>7.4f} {r['F1@5%']:>7.4f} {r['Best-F1']:>7.4f} {r['D']:>7.4f}")

summary = {"dataset":"HumanEval","model":MODEL,"max_samples":N,"results":all_results}
with open(f"{LOG_DIR}/summary_detection_v3.json","w",encoding="utf-8") as f:
    json.dump(summary,f,ensure_ascii=False,indent=2)
print(f"\nSaved: {LOG_DIR}/summary_detection_v3.json\nDone.")
