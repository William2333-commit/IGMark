"""Prompt-agnostic detection: original gen × generic/empty/original prompt detection.

Existing watermarked texts (generated with diverse original prompts).
Detection: replace prefix context with generic prompt → test real prompt loss.
"""
import os, sys, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch, numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PID = os.getpid()
MODEL = "bigcode/starcoder"; N = 164
GENERIC = 'def solution(*args):\n    """Generate a solution\n    """\n'
GAMMA=0.5; PREFIX=1; HASH=15485863; TAU=1.0; EPS=1e-12; DREF=2.0
FOLDER="results/humaneval/compare"

print(f"[PID={PID}] Prompt loss: original gen × generic/empty/original detection")

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, trust_remote_code=True).to("cuda:0")
model.eval()

# Load existing texts (generated with original, diverse prompts)
igsw_file = next(f for f in os.listdir(FOLDER) if f.startswith("igsw_k12") and f.endswith(".jsonl"))
with open(f"{FOLDER}/{igsw_file}") as f:
    bdata = [json.loads(l) for l in f][:N]
non_texts = [d["unwatermarked_text"] for d in bdata]
orig_prompts = [d["prompt"] for d in bdata]

wm_texts = {}
wm_prompts = {}
for alg, fn in {"IGSW":"igsw_k12.jsonl","KGW":"kgw.jsonl","SWEET":"sweet.jsonl","DBW":"dbw.jsonl"}.items():
    with open(f"{FOLDER}/{fn}") as f:
        data = [json.loads(l) for l in f][:N]
    wm_texts[alg] = [d["watermarked_text"] for d in data]
    wm_prompts[alg] = [d.get("prompt","") for d in data]

# ========== Utilities ==========
def get_gl(prefix_ids, device, vs):
    tr = 1
    for t in prefix_ids: tr *= int(t)
    rng = torch.Generator(device=device); rng.manual_seed(HASH*(tr%vs))
    return torch.randperm(vs, device=device, generator=rng)[:int(vs*GAMMA)]

def spike_entropy(probs, tau):
    return float(torch.sum(probs/(1.0+tau*probs)).cpu())

def ig_from_gm(g, d, ep=1e-12):
    g = max(0.0, min(1.0, float(g))); a = math.exp(d); z = 1.0+(a-1.0)*g
    return float(max((a*g)/max(z,ep)*d - math.log(max(z,ep)), 0.0))

def linear_w(xs, ep=1e-12):
    x = np.array(xs, dtype=float)
    if len(x)==0: return x
    w = np.maximum(x-float(np.min(x)), 0.0)
    return np.ones_like(x) if float(np.sum(w*w))<=ep else w.astype(float)

@torch.no_grad()
def compute_features(text, prompt):
    if not text: return None
    dev = next(model.parameters()).device
    pids = tokenizer.encode(prompt, add_special_tokens=False) if prompt else []
    cids = tokenizer.encode(text, add_special_tokens=False)
    if len(cids)==0: return None
    fids = pids + cids; sp = len(pids)
    logits = model(input_ids=torch.tensor([fids[:-1]], dtype=torch.long, device=dev)).logits[0]
    gf, se, igs = [], [], []
    for pos in range(sp+PREFIX, len(fids)):
        if pos-1>=len(logits): continue
        prefix = fids[max(0,pos-PREFIX):pos]
        probs = torch.softmax(logits[pos-1].float(), dim=-1)
        gl = get_gl(prefix, dev, len(tokenizer))
        gf.append(1.0 if (gl==fids[pos]).any().cpu().item() else 0.0)
        gm = float(torch.sum(probs[gl]).cpu())
        se.append(spike_entropy(probs, TAU))
        igs.append(ig_from_gm(gm, DREF))
    return {"green":np.array(gf),"entropy":np.array(se),"ig":np.array(igs)}

def score_ewd(f):
    g=f["green"]; se=f["entropy"]
    if len(g)==0: return None
    w=linear_w(se,EPS); sw=float(np.sum(w)); sw2=float(np.sum(w*w))
    if sw2<EPS: return None
    return float((np.sum(w*g)-GAMMA*sw)/math.sqrt(GAMMA*(1.0-GAMMA)*sw2))

def score_igwd(f):
    g=f["green"]; ig=f["ig"]
    if len(g)==0: return None
    w=linear_w(ig,EPS); sw=float(np.sum(w)); sw2=float(np.sum(w*w))
    if sw2<EPS: return None
    return float((np.sum(w*g)-GAMMA*sw)/math.sqrt(GAMMA*(1.0-GAMMA)*sw2))

def compute_metrics(ws, ns):
    if len(ws)==0: return {"AUROC":0,"TPR@1%":0,"F1@1%":0,"TPR@5%":0,"F1@5%":0,"Best-F1":0,"D":0,"pairs":0}
    yt=np.array([1]*len(ws)+[0]*len(ns)); ys=np.array(ws+ns)
    auroc=float(roc_auc_score(yt,ys))
    fpr,tpr,ths=roc_curve(yt,ys)
    def mt(t):
        valid=np.where(fpr<=t)[0]
        if len(valid)==0: return 0.0,0.0
        idx=valid[np.argmax(tpr[valid])]
        tp=int(np.sum(np.array(ws)>=ths[idx])); fp=int(np.sum(np.array(ns)>=ths[idx]))
        fn=len(ws)-tp
        prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1)
        return float(rec),float(2*prec*rec/max(prec+rec,1e-12))
    tpr1,f1_1=mt(0.01); tpr5,f1_5=mt(0.05)
    bf=0.0
    for th in sorted(set(ys),reverse=True):
        tp=int(np.sum(np.array(ws)>=th)); fp=int(np.sum(np.array(ns)>=th))
        fn=len(ws)-tp
        prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1)
        f1=2*prec*rec/max(prec+rec,1e-12)
        if f1>bf: bf=f1
    D=round(float((auroc+tpr5)/2),4)
    return {"AUROC":round(auroc,4),"TPR@1%":round(tpr1,4),"F1@1%":round(f1_1,4),
            "TPR@5%":round(tpr5,4),"F1@5%":round(f1_5,4),"Best-F1":round(bf,4),
            "D":D,"pairs":len(ws),"wm_z":round(float(np.mean(ws)),2),"non_z":round(float(np.mean(ns)),2)}

# ========== Three detection modes ==========
MODES = {
    "original": lambda i,alg: orig_prompts[i] if alg=="non" else wm_prompts[alg][i],
    "generic":  lambda i,alg: GENERIC,
    "empty":    lambda i,alg: "",
}

DETECTORS = ["EWD","IGWD"]
all_results = []

for mode_name, prompt_fn in MODES.items():
    print(f"\n{'='*50}\nDetection prefix: {mode_name}\n{'='*50}")

    # Precompute non features
    non_feat = [compute_features(non_texts[i], prompt_fn(i,"non")) for i in tqdm(range(N), desc="non")]
    non_igwd = [s for f in non_feat if f is not None and (s:=score_igwd(f)) is not None]
    non_ewd  = [s for f in non_feat if f is not None and (s:=score_ewd(f))  is not None]

    for alg in ["IGSW","SWEET","KGW","DBW"]:
        texts = wm_texts[alg]
        wm_feat = [compute_features(texts[i], prompt_fn(i,alg)) for i in tqdm(range(N), desc=f"{alg} feat")]

        for det in DETECTORS:
            if det == "IGWD":
                wm_z = [s for f in wm_feat if f is not None and (s:=score_igwd(f)) is not None]
                non_z = non_igwd
            else:
                wm_z = [s for f in wm_feat if f is not None and (s:=score_ewd(f)) is not None]
                non_z = non_ewd

            n = min(len(wm_z), len(non_z)); m = compute_metrics(wm_z[:n], non_z[:n])
            print(f"  {alg}+{det}: D={m['D']:.4f} AUROC={m['AUROC']:.4f} wm_z={m['wm_z']:.2f} non_z={m['non_z']:.2f}")
            all_results.append({"algorithm":alg,"prompt_mode":mode_name,"detection":det,**m})

# Summary
print(f"\n{'='*105}")
print(f"{'Alg':<6} {'Det':<5} {'Prefix':<10} {'AUROC':>7} {'TPR5':>7} {'D':>7} {'wm_z':>7} {'D_decay':>9}")
print("-"*105)

base_d = {}
for r in all_results:
    if r['prompt_mode']=='original': base_d[(r['algorithm'],r['detection'])] = r['D']

for r in all_results:
    base = base_d.get((r['algorithm'],r['detection']), 1)
    decay = f"{(base-r['D'])/base*100:.0f}%" if base>0 else "-"
    print(f"{r['algorithm']:<6} {r['detection']:<5} {r['prompt_mode']:<10} {r['AUROC']:>7.4f} {r['TPR@5%']:>7.4f} {r['D']:>7.4f} {r['wm_z']:>7.2f} {decay:>9}")

os.makedirs("results/generic_prompt", exist_ok=True)
with open("results/generic_prompt/prompt_loss.json","w") as f:
    json.dump({"dataset":"HumanEval","generic_prompt":GENERIC,"N":N,"results":all_results}, f, ensure_ascii=False, indent=2)
print(f"\nSaved: results/generic_prompt/prompt_loss.json\nDone!")
