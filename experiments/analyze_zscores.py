"""Z-score distribution: unified KGW z-test for fair cross-algorithm comparison.

Methods: all use KGW standard z-test, except EWD on KGW as reference.
"""
import os, sys, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch, numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

TARGET_DS = sys.argv[1] if len(sys.argv) > 1 else None
GAMMA=0.5; PREFIX=1; TAU=1.0; EPS=1e-12

CONFIGS = {
    "HumanEval": {"model":"bigcode/starcoder","folder":"results/humaneval/compare","baseline_key":"unwatermarked_text","baseline_file":"igsw_k12.jsonl"},
    "WT2": {"model":"facebook/opt-1.3b","folder":"results/wt2/sample_t0.7","baseline_key":"unwatermarked_text","baseline_file":"baseline.jsonl"},
}

def parse_score(r):
    if isinstance(r,dict):
        for k in ["score","z_score","z"]:
            if k in r: return float(r[k])
    if isinstance(r,(int,float)): return float(r)
    return None

def safe_detect(wm,text):
    if not text or len(text.strip())==0: return None
    try:
        s=parse_score(wm.detect_watermark(text,return_dict=True))
        return s if s is not None and not math.isnan(s) and not math.isinf(s) else None
    except: return None

# ============ EWD detector (for KGW texts, as comparison) ============
def get_utils(w):
    for n in ["utils","watermark_utils","algorithm_utils"]:
        if hasattr(w,n): return getattr(w,n)
def get_gl(wm,prefix_ids,device):
    utils=get_utils(wm)
    for x in [torch.tensor(prefix_ids,dtype=torch.long,device=device),torch.tensor(prefix_ids,dtype=torch.long,device="cpu"),prefix_ids]:
        try:
            ids=utils.get_greenlist_ids(x)
            if isinstance(ids,tuple): ids=ids[0]
            if isinstance(ids,list): ids=torch.tensor(ids,dtype=torch.long,device=device)
            elif isinstance(ids,np.ndarray): ids=torch.tensor(ids,dtype=torch.long,device=device)
            elif torch.is_tensor(ids): ids=ids.to(device)
            else: ids=torch.tensor(list(ids),dtype=torch.long,device=device)
            return ids.view(-1)
        except: pass
    raise RuntimeError

def spike_entropy(probs,tau): return float(torch.sum(probs/(1.0+tau*probs)).cpu())
def linear_w(xs,ep=1e-12):
    x=np.array(xs,dtype=float)
    if len(x)==0: return x
    c0=float(np.min(x)); w=np.maximum(x-c0,0.0)
    return np.ones_like(x) if float(np.sum(w*w))<=ep else w.astype(float)

@torch.no_grad()
def compute_features(wm,prompt,text,tokenizer,model):
    if not text or len(text.strip())==0: return {"green":[],"entropy":[]}
    dev=next(model.parameters()).device
    pids=tokenizer.encode(prompt,add_special_tokens=False) if prompt else []
    cids=tokenizer.encode(text,add_special_tokens=False)
    if len(cids)==0: return {"green":[],"entropy":[]}
    fids=pids+cids
    if len(fids)<2: return {"green":[],"entropy":[]}
    logits=model(input_ids=torch.tensor([fids[:-1]],dtype=torch.long,device=dev)).logits[0]
    sp=len(pids); gf,se=[],[]
    for pos in range(sp,len(fids)):
        if pos<=0 or pos-1>=len(logits): continue
        prefix=fids[max(0,pos-PREFIX):pos]
        if len(prefix)==0: continue
        probs=torch.softmax(logits[pos-1].float(),dim=-1)
        gids=get_gl(wm,prefix,probs.device)
        gf.append(1.0 if bool((gids==fids[pos]).any().cpu().item()) else 0.0)
        se.append(spike_entropy(probs,TAU))
    return {"green":gf,"entropy":se}

def score_ewd(f):
    g=f["green"]; se=f["entropy"]
    if len(g)==0: return None
    w=linear_w(se,EPS)
    denom=GAMMA*(1.0-GAMMA)*float(np.sum(w*w))
    return float((np.sum(w*np.array(g))-GAMMA*float(np.sum(w)))/math.sqrt(denom)) if denom>EPS else None

# ============ Main ============
results={}

for ds_name,ds_cfg in CONFIGS.items():
    if TARGET_DS and ds_name!=TARGET_DS: continue
    print(f"\n{'='*60}\n{ds_name}\n{'='*60}")
    model_name=ds_cfg["model"]
    if 'model' in dir(): del model; torch.cuda.empty_cache()

    print(f"Loading {model_name}...")
    tokenizer=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16,trust_remote_code=True).to("cuda:0")
    model.eval()
    kw = {"max_new_tokens":512,"do_sample":True,"temperature":0.2,"top_p":0.95} if "starcoder" in model_name else {"max_new_tokens":512,"do_sample":True,"temperature":0.7,"top_p":0.95}
    tcfg=TransformersConfig(model=model,tokenizer=tokenizer,vocab_size=len(tokenizer),device="cuda",**kw)

    # Load baseline
    with open(os.path.join(ds_cfg["folder"],ds_cfg["baseline_file"])) as f:
        bdata=[json.loads(l) for l in f]
    non_texts=[d[ds_cfg["baseline_key"]] for d in bdata]
    non_prompts=[d.get("prompt","") for d in bdata]
    N=min(len(non_texts),500)
    print(f"  Non texts: {len(non_texts)} → use {N}")
    folder=ds_cfg["folder"]

    # === Unified KGW detector ===
    wm_kgw=AutoWatermark.load("KGW",algorithm_config="config/KGW.json",transformers_config=tcfg)
    ds_results={}

    # 1. KGW z-test on IGSW texts
    print("  [1] KGW on IGSW")
    igsw_file=next(f for f in os.listdir(folder) if f.startswith("igsw") and f.endswith(".jsonl"))
    with open(os.path.join(folder,igsw_file)) as f:
        igsw_data=[json.loads(l) for l in f]
    igsw_texts=[d.get("watermarked_text","") for d in igsw_data][:N]
    wm_zs=[s for t in tqdm(igsw_texts,desc="IGSW wm") if (s:=safe_detect(wm_kgw,t)) is not None]
    non_zs=[s for t in tqdm(non_texts[:N],desc="non") if (s:=safe_detect(wm_kgw,t)) is not None]
    n=min(len(wm_zs),len(non_zs))
    ds_results["IGSW"]={"wm":wm_zs[:n],"non":non_zs[:n]}
    print(f"    wm_z={np.mean(wm_zs[:n]):.2f} non_z={np.mean(non_zs[:n]):.2f}  pairs={n}")

    # 2. KGW z-test on SWEET texts
    print("  [2] KGW on SWEET")
    with open(os.path.join(folder,"sweet.jsonl")) as f:
        sweet_data=[json.loads(l) for l in f]
    sweet_texts=[d.get("watermarked_text","") for d in sweet_data][:N]
    wm_zs=[s for t in tqdm(sweet_texts,desc="SWEET wm") if (s:=safe_detect(wm_kgw,t)) is not None]
    non_zs=[s for t in tqdm(non_texts[:N],desc="non") if (s:=safe_detect(wm_kgw,t)) is not None]
    n=min(len(wm_zs),len(non_zs))
    ds_results["SWEET"]={"wm":wm_zs[:n],"non":non_zs[:n]}
    print(f"    wm_z={np.mean(wm_zs[:n]):.2f} non_z={np.mean(non_zs[:n]):.2f}  pairs={n}")

    # 3. KGW z-test on DBW texts
    print("  [3] KGW on DBW")
    with open(os.path.join(folder,"dbw.jsonl")) as f:
        dbw_data=[json.loads(l) for l in f]
    dbw_texts=[d.get("watermarked_text","") for d in dbw_data][:N]
    wm_zs=[s for t in tqdm(dbw_texts,desc="DBW wm") if (s:=safe_detect(wm_kgw,t)) is not None]
    non_zs=[s for t in tqdm(non_texts[:N],desc="non") if (s:=safe_detect(wm_kgw,t)) is not None]
    n=min(len(wm_zs),len(non_zs))
    ds_results["DBW"]={"wm":wm_zs[:n],"non":non_zs[:n]}
    print(f"    wm_z={np.mean(wm_zs[:n]):.2f} non_z={np.mean(non_zs[:n]):.2f}  pairs={n}")

    # 4. KGW z-test on KGW texts (reference)
    print("  [4] KGW on KGW")
    with open(os.path.join(folder,"kgw.jsonl")) as f:
        kgw_data=[json.loads(l) for l in f]
    kgw_texts=[d.get("watermarked_text","") for d in kgw_data][:N]
    wm_zs=[s for t in tqdm(kgw_texts,desc="KGW wm") if (s:=safe_detect(wm_kgw,t)) is not None]
    non_zs=[s for t in tqdm(non_texts[:N],desc="non") if (s:=safe_detect(wm_kgw,t)) is not None]
    n=min(len(wm_zs),len(non_zs))
    ds_results["KGW"]={"wm":wm_zs[:n],"non":non_zs[:n]}
    print(f"    wm_z={np.mean(wm_zs[:n]):.2f} non_z={np.mean(non_zs[:n]):.2f}  pairs={n}")

    # 5. EWD on KGW texts (comparison)
    print("  [5] EWD on KGW")
    kgw_prompts=[d.get("prompt","") for d in kgw_data][:N]
    wm_feats=[compute_features(wm_kgw,kgw_prompts[i],kgw_texts[i],tokenizer,model) for i in tqdm(range(len(kgw_texts)),desc="KGW feat")]
    non_feats=[compute_features(wm_kgw,non_prompts[i],non_texts[i],tokenizer,model) for i in tqdm(range(N),desc="non feat")]
    wm_zs=[s for f in wm_feats if (s:=score_ewd(f)) is not None]
    non_zs=[s for f in non_feats if (s:=score_ewd(f)) is not None]
    n=min(len(wm_zs),len(non_zs))
    ds_results["KGW+EWD"]={"wm":wm_zs[:n],"non":non_zs[:n]}
    print(f"    wm_z={np.mean(wm_zs[:n]):.2f} non_z={np.mean(non_zs[:n]):.2f}  pairs={n}")

    results[ds_name]=ds_results

# Save
os.makedirs("results/analysis",exist_ok=True)
outname=f"results/analysis/zscores_{TARGET_DS}.json" if TARGET_DS else "results/analysis/zscores.json"
save={ds:{alg:{"wm":[float(x) for x in data["wm"]],"non":[float(x) for x in data["non"]]} for alg,data in algs.items()} for ds,algs in results.items()}
with open(outname,"w") as f: json.dump(save,f)
print(f"\nSaved: {outname}\nDone.")
