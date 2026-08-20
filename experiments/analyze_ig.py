"""IG distribution analysis across 4 datasets.

Computes per-token IG, green_mass, spike_entropy from unwatermarked texts.
Generates: IG histogram, green_mass vs SE scatter, delta mapping overlay.
"""
import os, sys, json, math, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch, numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

GAMMA = 0.5; PREFIX = 1; DREF = 2.0; TAU = 1.0; HASH = 15485863

DATASETS = {
    "HumanEval": {
        "model": "bigcode/starcoder",
        "baseline": "results/humaneval/compare/igsw_k12.jsonl",
        "key": "unwatermarked_text",
    },
    "MBPP": {
        "model": "bigcode/starcoder",
        "baseline_pkl": "results/mbpp/shared_baseline/baseline.pkl",
        "prompt_jsonl": "results/mbpp/compare/igsw_k=12.jsonl",
    },
    "WMT": {
        "model": "facebook/opt-1.3b",
        "baseline": "results/wmt16/sample_t0.7/baseline.jsonl",
        "key": "unwatermarked_text",
    },
    "C4": {
        "model": "facebook/opt-1.3b",
        "baseline": "results/c4/gen/baseline.jsonl",
        "key": "unwatermarked_text",
    },
}

def spike_entropy(probs, tau):
    return float(torch.sum(probs / (1.0 + tau * probs)).cpu())

def ig_from_green_mass(g, d, eps=1e-12):
    g = max(0.0, min(1.0, float(g)))
    a = math.exp(float(d)); z = 1.0 + (a - 1.0) * g
    return float(max((a*g)/max(z,eps)*float(d) - math.log(max(z,eps)), 0.0))

@torch.no_grad()
def analyze_dataset(name, cfg, model, tokenizer):
    print(f"\n=== {name} ===")
    dev = next(model.parameters()).device

    # Load texts
    if "baseline_pkl" in cfg:
        with open(cfg["baseline_pkl"], "rb") as f:
            bd = pickle.load(f)
        non_texts = bd["non_texts"]
        with open(cfg["prompt_jsonl"]) as f:
            prompts = [json.loads(l)["prompt"] for l in f]
    else:
        with open(cfg["baseline"]) as f:
            data = [json.loads(l) for l in f]
        non_texts = [d[cfg["key"]] for d in data]
        prompts = [d.get("prompt", "") for d in data]

    N = min(len(non_texts), 2000)
    all_ig, all_gm, all_se = [], [], []

    for i in tqdm(range(N), desc=name):
        text = non_texts[i]
        prompt = prompts[i] if i < len(prompts) else ""
        if not text or len(text.strip()) == 0:
            continue

        pids = tokenizer.encode(prompt, add_special_tokens=False) if prompt else []
        cids = tokenizer.encode(text, add_special_tokens=False)
        fids = pids + cids
        if len(fids) < 2:
            continue

        # Forward pass on text only (or prompt+text)
        logits = model(input_ids=torch.tensor([fids[:-1]], dtype=torch.long, device=dev)).logits[0]
        sp = len(pids)

        for pos in range(sp, len(fids)):
            if pos <= 0:
                continue
            prefix = fids[max(0, pos - PREFIX):pos]
            if len(prefix) == 0:
                continue
            probs = torch.softmax(logits[pos - 1].float(), dim=-1)
            # Greenlist
            time_result = 1
            for t in prefix:
                time_result *= int(t)
            rng = torch.Generator(device=dev)
            rng.manual_seed(HASH * (time_result % len(tokenizer)))
            gsize = int(len(tokenizer) * GAMMA)
            gl_ids = torch.randperm(len(tokenizer), device=dev, generator=rng)[:gsize]
            gm = float(torch.sum(probs[gl_ids]).cpu())
            ig = ig_from_green_mass(gm, DREF)
            se = spike_entropy(probs, TAU)
            all_ig.append(ig)
            all_gm.append(gm)
            all_se.append(se)

    ig = np.array(all_ig); gm = np.array(all_gm); se = np.array(all_se)
    print(f"  Tokens: {len(ig)}")
    print(f"  IG:      mean={ig.mean():.4f} std={ig.std():.4f} min={ig.min():.4f} max={ig.max():.4f}")
    print(f"  GM:      mean={gm.mean():.4f} std={gm.std():.4f}")
    print(f"  SE:      mean={se.mean():.4f} std={se.std():.4f}")
    print(f"  IG<0.15: {(ig < 0.15).mean()*100:.1f}%  IG>0.5: {(ig > 0.5).mean()*100:.1f}%")

    # Delta distribution (tanh mapping with c0=0.15, k=12, d_min=1.0, d_max=3.0)
    k, c0, dmin, dmax = 12, 0.15, 1.0, 3.0
    delta = dmax - (dmax - dmin) / (1 + np.exp(-k * (ig - c0)))
    print(f"  δ:       mean={delta.mean():.2f} δ>2.0: {(delta > 2.0).mean()*100:.1f}% δ>2.5: {(delta > 2.5).mean()*100:.1f}%")
    print(f"  δ bins:  <1.5:{(delta<1.5).mean()*100:.0f}%  1.5-2.0:{(np.abs(delta-1.75)<0.25).mean()*100:.0f}%  2.0-2.5:{(np.abs(delta-2.25)<0.25).mean()*100:.0f}%  >2.5:{(delta>2.5).mean()*100:.0f}%")

    return {"ig": ig, "gm": gm, "se": se, "delta": delta}

# Main
results = {}
for name, cfg in DATASETS.items():
    model_name = cfg["model"]
    print(f"\nLoading {model_name} for {name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    # Free previous model, load new
    if 'model' in dir():
        del model; torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True).to("cuda:0")
    model.eval()
    results[name] = analyze_dataset(name, cfg, model, tokenizer)

# Summary table
print(f"\n{'='*100}")
print(f"{'Dataset':<14} {'Model':<16} {'Tokens':>8} {'IG mean':>8} {'IG<0.15%':>9} {'δ>2.0%':>8} {'δ>2.5%':>8} {'δ mean':>8}")
print("-"*100)
for name, r in results.items():
    cfg = DATASETS[name]
    ig = r["ig"]; delta = r["delta"]
    print(f"{name:<14} {cfg['model']:<16} {len(ig):>8} {ig.mean():>8.4f} {((ig<0.15).mean()*100):>9.1f} {(delta>2.0).mean()*100:>8.1f} {(delta>2.5).mean()*100:>8.1f} {delta.mean():>8.2f}")

# Save for plotting
os.makedirs("results/analysis", exist_ok=True)
for name, r in results.items():
    np.savez(f"results/analysis/{name}_ig.npz", ig=r["ig"], gm=r["gm"], se=r["se"], delta=r["delta"])
print("\nSaved: results/analysis/*_ig.npz")
