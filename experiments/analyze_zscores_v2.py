"""Z-score analysis v2: IGSW+IGWD with KGW-style γ-based z-test + prompt context.

IGSW: z = (Σg - γT) / √(γ(1-γ)T)  — same formula as KGW/DBW
IGWD: z = (Σw·g - γΣw) / √(γ(1-γ)Σw²), w = max(max(IG)-IG, 0) — flipped weight

Other methods: built-in detect_watermark()
"""
import os, sys, json, math, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch, numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

TARGET_DS = sys.argv[1] if len(sys.argv) > 1 else None
if not TARGET_DS:
    print("Usage: python analyze_zscores_v2.py HumanEval|WT2")
    sys.exit(1)

GAMMA=0.5; PREFIX=1; HASH=15485863; DREF=2.0
IGSW_CFG={"algorithm_name":"IGSW","gamma":0.5,"delta_ref":2.0,"hash_key":HASH,
    "z_threshold":4.0,"prefix_length":1,"eps":1e-12,"visualize_mode":"raw_ig",
    "function":"tanh","delta_min":1.0,"delta_max":3.0,"c0":0.15,"k":12}

CONFIGS={
    "HumanEval":{"model":"bigcode/starcoder","folder":"results/humaneval/compare",
        "baseline_key":"unwatermarked_text","baseline_file":"igsw_k12.jsonl"},
    "WT2":{"model":"facebook/opt-1.3b","folder":"results/wt2/sample_t0.7",
        "baseline_key":"unwatermarked_text","baseline_file":"baseline.jsonl"},
}

ds_cfg=CONFIGS[TARGET_DS]
print(f"[{TARGET_DS}] Loading {ds_cfg['model']}...")
tokenizer=AutoTokenizer.from_pretrained(ds_cfg["model"],trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
model=AutoModelForCausalLM.from_pretrained(ds_cfg["model"],torch_dtype=torch.float16,trust_remote_code=True).to("cuda:0")
model.eval()

# Load baseline
with open(os.path.join(ds_cfg["folder"],ds_cfg["baseline_file"])) as f:
    bdata=[json.loads(l) for l in f]
non_texts=[d[ds_cfg["baseline_key"]] for d in bdata]
non_prompts=[d.get("prompt","") for d in bdata]
N=min(len(non_texts),500)
print(f"Non texts: {len(non_texts)} -> {N}")
folder=ds_cfg["folder"]

# ============ Greenlist using KGW hash ============
def get_gl_manual(prefix_ids, device, vocab_size):
    time_result=1
    for t in prefix_ids: time_result*=int(t)
    rng=torch.Generator(device=device)
    rng.manual_seed(HASH*(time_result%vocab_size))
    gsize=int(vocab_size*GAMMA)
    return torch.randperm(vocab_size,device=device,generator=rng)[:gsize]

# ============ KGW-style γ-based z-test (for IGSW) ============
def kgw_zscore(text, prompt=""):
    """Standard KGW z-test: z = (|G|-γT)/√(γ(1-γ)T) with prompt context."""
    if not text or len(text.strip())==0: return None
    dev=next(model.parameters()).device
    pids=tokenizer.encode(prompt,add_special_tokens=False) if prompt else []
    cids=tokenizer.encode(text,add_special_tokens=False)
    if len(cids)==0: return None
    fids=pids+cids; sp=len(pids)
    green_count,total=0,0
    for pos in range(sp+PREFIX, len(fids)):
        prefix=fids[max(0,pos-PREFIX):pos]
        gl=get_gl_manual(prefix,dev,len(tokenizer))
        if (gl==fids[pos]).any().cpu().item(): green_count+=1
        total+=1
    if total==0: return None
    numer=green_count-GAMMA*total
    denom=math.sqrt(GAMMA*(1.0-GAMMA)*total)
    return float(numer/denom) if denom>1e-12 else None

# ============ IGWD with flipped weight ============
def compute_ig_and_green(text, prompt=""):
    """Compute per-token IG and green flag with prompt context."""
    if not text or len(text.strip())==0: return None
    dev=next(model.parameters()).device
    pids=tokenizer.encode(prompt,add_special_tokens=False) if prompt else []
    cids=tokenizer.encode(text,add_special_tokens=False)
    if len(cids)==0: return None
    fids=pids+cids; sp=len(pids)
    gf,igs=[],[]
    logits=model(input_ids=torch.tensor([fids[:-1]],dtype=torch.long,device=dev)).logits[0]
    for pos in range(sp+PREFIX, len(fids)):
        if pos-1>=len(logits): continue
        prefix=fids[max(0,pos-PREFIX):pos]
        probs=torch.softmax(logits[pos-1].float(),dim=-1)
        gl=get_gl_manual(prefix,dev,len(tokenizer))
        gf.append(1.0 if (gl==fids[pos]).any().cpu().item() else 0.0)
        gm=float(torch.sum(probs[gl]).cpu())
        a=math.exp(DREF); z=1.0+(a-1.0)*gm
        ig=float(max((a*gm)/max(z,1e-12)*DREF-math.log(max(z,1e-12)),0.0))
        igs.append(ig)
    return {"green":np.array(gf),"ig":np.array(igs)}

def igwd_zscore(text, prompt=""):
    """IGWD with flipped weight: w=max(max(IG)-IG,0), γ-based z."""
    f=compute_ig_and_green(text,prompt)
    if f is None or len(f["ig"])==0: return None
    ig=f["ig"]; gf=f["green"]
    max_ig=max(ig); w=np.maximum(max_ig-ig,0.0)  # flipped!
    sw=float(np.sum(w)); sw2=float(np.sum(w*w))
    if sw2<1e-12: return None
    return float((np.sum(w*gf)-GAMMA*sw)/math.sqrt(GAMMA*(1.0-GAMMA)*sw2))

# ============ Built-in detector for other methods ============
def parse_score(r):
    if isinstance(r,dict):
        for k in ["score","z_score","z"]:
            if k in r: return float(r[k])
    if isinstance(r,(int,float)): return float(r)
    return None

def safe_detect_builtin(wm,text):
    if not text or len(text.strip())==0: return None
    try:
        s=parse_score(wm.detect_watermark(text,return_dict=True))
        return s if s is not None and not math.isnan(s) else None
    except: return None

results={}

# Load watermarks once
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
kw={"max_new_tokens":512,"do_sample":True,"temperature":0.2,"top_p":0.95} if "starcoder" in ds_cfg["model"] else {"max_new_tokens":512,"do_sample":True,"temperature":0.7,"top_p":0.95}
tcfg=TransformersConfig(model=model,tokenizer=tokenizer,vocab_size=len(tokenizer),device="cuda",**kw)

print("Loading watermarks...")
wm_kgw=AutoWatermark.load("KGW",algorithm_config="config/KGW.json",transformers_config=tcfg)
wm_dbw=AutoWatermark.load("DBW",algorithm_config="config/DBW.json",transformers_config=tcfg)

# SWEET 0.65
tmp=tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False)
json.dump({"algorithm_name":"SWEET","gamma":0.5,"delta":2.0,"hash_key":HASH,"z_threshold":4.0,"prefix_length":1,"entropy_threshold":0.65},tmp); tmp.close()
wm_sweet=AutoWatermark.load("SWEET",algorithm_config=tmp.name,transformers_config=tcfg)
os.unlink(tmp.name)

# Load text files
igsw_file=next(f for f in os.listdir(folder) if f.startswith("igsw") and f.endswith(".jsonl"))
with open(os.path.join(folder,igsw_file)) as f: igsw_data=[json.loads(l) for l in f]
with open(os.path.join(folder,"kgw.jsonl")) as f: kgw_data=[json.loads(l) for l in f]
with open(os.path.join(folder,"sweet.jsonl")) as f: sweet_data=[json.loads(l) for l in f]
with open(os.path.join(folder,"dbw.jsonl")) as f: dbw_data=[json.loads(l) for l in f]

igsw_texts=[d.get("watermarked_text","") for d in igsw_data][:N]
igsw_prompts=[d.get("prompt","") for d in igsw_data][:N]
kgw_texts=[d.get("watermarked_text","") for d in kgw_data][:N]
kgw_prompts=[d.get("prompt","") for d in kgw_data][:N]
sweet_texts=[d.get("watermarked_text","") for d in sweet_data][:N]
dbw_texts=[d.get("watermarked_text","") for d in dbw_data][:N]

# 1. IGSW (KGW-style γ-based z-test)
print("\n[1] IGSW (KGW z-test)")
wm_zs=[s for i in tqdm(range(N),desc="IGSW wm") if (s:=kgw_zscore(igsw_texts[i],igsw_prompts[i])) is not None]
non_zs=[s for i in tqdm(range(N),desc="non") if (s:=kgw_zscore(non_texts[i],non_prompts[i])) is not None]
n=min(len(wm_zs),len(non_zs))
results["IGSW"]={"wm":wm_zs[:n],"non":non_zs[:n]}
print(f"  wm_z={np.mean(wm_zs[:n]):.2f} non_z={np.mean(non_zs[:n]):.2f} pairs={n}")

# 2. IGWD (flipped weight, γ-based z-test)
print("\n[2] IGWD (flipped weight)")
wm_zs=[s for i in tqdm(range(N),desc="IGSW wm") if (s:=igwd_zscore(igsw_texts[i],igsw_prompts[i])) is not None]
non_zs=[s for i in tqdm(range(N),desc="non") if (s:=igwd_zscore(non_texts[i],non_prompts[i])) is not None]
n=min(len(wm_zs),len(non_zs))
results["IGSW+IGWD"]={"wm":wm_zs[:n],"non":non_zs[:n]}
print(f"  wm_z={np.mean(wm_zs[:n]):.2f} non_z={np.mean(non_zs[:n]):.2f} pairs={n}")

# 3. KGW built-in
print("\n[3] KGW built-in")
wm_zs=[s for i in tqdm(range(N),desc="KGW wm") if (s:=safe_detect_builtin(wm_kgw,kgw_texts[i])) is not None]
non_zs=[s for i in tqdm(range(N),desc="non") if (s:=safe_detect_builtin(wm_kgw,non_texts[i])) is not None]
n=min(len(wm_zs),len(non_zs))
results["KGW"]={"wm":wm_zs[:n],"non":non_zs[:n]}
print(f"  wm_z={np.mean(wm_zs[:n]):.2f} non_z={np.mean(non_zs[:n]):.2f} pairs={n}")

# 4. DBW built-in
print("\n[4] DBW built-in")
wm_zs=[s for i in tqdm(range(N),desc="DBW wm") if (s:=safe_detect_builtin(wm_dbw,dbw_texts[i])) is not None]
non_zs=[s for i in tqdm(range(N),desc="non") if (s:=safe_detect_builtin(wm_dbw,non_texts[i])) is not None]
n=min(len(wm_zs),len(non_zs))
results["DBW"]={"wm":wm_zs[:n],"non":non_zs[:n]}
print(f"  wm_z={np.mean(wm_zs[:n]):.2f} non_z={np.mean(non_zs[:n]):.2f} pairs={n}")

# 5. SWEET built-in
print("\n[5] SWEET built-in")
wm_zs=[s for i in tqdm(range(N),desc="SWEET wm") if (s:=safe_detect_builtin(wm_sweet,sweet_texts[i])) is not None]
non_zs=[s for i in tqdm(range(N),desc="non") if (s:=safe_detect_builtin(wm_sweet,non_texts[i])) is not None]
n=min(len(wm_zs),len(non_zs))
results["SWEET"]={"wm":wm_zs[:n],"non":non_zs[:n]}
print(f"  wm_z={np.mean(wm_zs[:n]):.2f} non_z={np.mean(non_zs[:n]):.2f} pairs={n}")

# Save
os.makedirs("results/analysis",exist_ok=True)
out=f"results/analysis/zscores_v2_{TARGET_DS}.json"
save={TARGET_DS:{alg:{"wm":[float(x) for x in d["wm"]],"non":[float(x) for x in d["non"]]} for alg,d in results.items()}}
with open(out,"w") as f: json.dump(save,f)
print(f"\nSaved: {out}\nDone.")
