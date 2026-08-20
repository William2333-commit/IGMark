"""固定 IGSW 生成端，比较 Uniform-g / IG-g / EWD-g 检测配置。

生成端统一 IGSW 动态 δ_t = tanh(IG_t)，r_t 用 IGSW 的 δ_t。
检测权重不同，零假设校准都用 g_t。

用法: CUDA_VISIBLE_DEVICES=2 python experiments/compute_igsw_ablation.py [--max_samples 164]
"""
import os, sys, json, math, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from scipy.stats import norm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from watermark.igsw.igsw import IGSWConfig, IGSWUtils
from utils.transformers_config import TransformersConfig

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

GAMMA = 0.5
PREFIX_LEN = 1
DELTA_REF = 2.0
# IGSW tanh
D_MIN, D_MAX, K_IG, C0 = 1.0, 3.0, 12, 0.15
# EWD spike entropy
ALPHA_EWD = math.exp(2.0)
Z_VALUE = ((1-GAMMA)*(ALPHA_EWD-1))/(1-GAMMA+ALPHA_EWD*GAMMA)
TAUS = {"tau2.0": 2.0, "tau1%": 2.3263, "tau5%": 1.6449}


def igsw_delta(ig):
    t = (math.tanh(K_IG * (ig - C0)) + 1.0) / 2.0
    return D_MIN + (D_MAX - D_MIN) * t

def spike_entropy(probs):
    return float((probs / (1.0 + Z_VALUE * probs)).sum())

def ig_from_g(g):
    alpha = math.exp(DELTA_REF)
    z = 1.0 + (alpha-1.0)*g
    return (alpha*g)/max(z,1e-12)*DELTA_REF - math.log(max(z,1e-12))

def r_t(g, delta):
    a = math.exp(delta)
    return (g*a)/(1.0+(a-1.0)*g)


def compute_rows(tokens, probs_list, utils):
    rows = []
    for idx in range(PREFIX_LEN, len(tokens)):
        probs_t = probs_list[idx]
        if probs_t is None: continue
        greenlist = utils.get_greenlist_ids(tokens[:idx])
        g = float(probs_t[greenlist].sum())
        ig = ig_from_g(g)
        delta_igsw = igsw_delta(ig)
        SE = spike_entropy(probs_t)
        rows.append({"g": g, "ig": ig, "delta": delta_igsw, "SE": SE})
    return rows


def system_compute(rows, weight_fn):
    """δ_t 用 IGSW 动态, q=g_t（所有配置）。"""
    g = np.array([r["g"] for r in rows])
    a = np.array([weight_fn(r) for r in rows])
    d = np.array([r["delta"] for r in rows])  # IGSW 动态 δ
    r = np.array([r_t(g[i], d[i]) for i in range(len(g))])
    q = g  # 所有配置用 g_t 校准
    mu0 = float((a*g).sum())
    sig0 = float((a*a*g*(1-g)).sum())
    mu0h = float((a*q).sum())
    sig0h = float((a*a*q*(1-q)).sum())
    mu1 = float((a*r).sum())
    sig1 = float((a*a*r*(1-r)).sum())
    return mu0, sig0, mu0h, sig0h, mu1, sig1


def theory_errors(mu0, sig0, mu0h, sig0h, mu1, sig1, tau):
    c = mu0h + tau * math.sqrt(max(sig0h, 1e-12))
    s0 = math.sqrt(max(sig0, 1e-12))
    s1 = math.sqrt(max(sig1, 1e-12))
    alpha = 1 - norm.cdf((c - mu0) / s0)
    beta = norm.cdf((c - mu1) / s1)
    return alpha, beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_samples", type=int, default=164)
    ap.add_argument("--wm_file", default="results/humaneval/compare/igsw_k12.jsonl")
    args = ap.parse_args()

    print(f"生成端: IGSW tanh(δ_min={D_MIN}, δ_max={D_MAX}, k={K_IG}, c0={C0})")
    print(f"校准: 所有配置用 g_t")
    print("Loading StarCoder (8-bit)...")
    tok = AutoTokenizer.from_pretrained("bigcode/starcoder", trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder",
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map="auto", trust_remote_code=True)
    model.eval()
    tcfg = TransformersConfig(model=model, tokenizer=tok, vocab_size=len(tok), device="cuda",
                              max_new_tokens=512, do_sample=True, temperature=0.2, top_p=0.95)
    utils = IGSWUtils(IGSWConfig("config/IGSW.json", tcfg))

    data = [json.loads(l) for l in open(args.wm_file)][:args.max_samples]
    print(f"Samples: {len(data)} from {args.wm_file}")

    configs = ["IGSW+Uniform-g", "IGSW+IG-g", "IGSW+EWD-g"]
    acc = {c: {"alpha": {t: [] for t in TAUS}, "beta": {t: [] for t in TAUS}} for c in configs}

    for i, d in enumerate(data):
        text = d.get("watermarked_text", d.get("watermarked", ""))
        if not text.strip(): continue
        encoded = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(model.device)
        if len(encoded) <= PREFIX_LEN: continue
        with torch.no_grad():
            probs_list = utils.calculate_probabilities(model, encoded)
        rows = compute_rows(encoded, probs_list, utils)
        if not rows: continue

        ig_min = min(r["ig"] for r in rows)
        se_min = min(r["SE"] for r in rows)

        w_fns = {
            "IGSW+Uniform-g": lambda r: 1.0,
            "IGSW+IG-g": lambda r: max(r["ig"] - ig_min, 0.0),
            "IGSW+EWD-g": lambda r: max(r["SE"] - se_min, 0.0),
        }
        for cfg in configs:
            mu0, sig0, mu0h, sig0h, mu1, sig1 = system_compute(rows, w_fns[cfg])
            for tau_name, tau in TAUS.items():
                alpha, beta = theory_errors(mu0, sig0, mu0h, sig0h, mu1, sig1, tau)
                acc[cfg]["alpha"][tau_name].append(alpha)
                acc[cfg]["beta"][tau_name].append(beta)
        if (i+1) % 20 == 0: print(f"  {i+1}/{len(data)}", flush=True)

    N = len(acc["IGSW+Uniform-g"]["alpha"]["tau2.0"])
    print(f"\n有效样本: {N}\n")

    print("="*70)
    print("固定 IGSW 生成端，比较检测配置（q=g_t 校准）")
    print("="*70)
    print(f"\n{'检测配置':<20}{'α@2%':>10}{'β@2%':>10}{'α@1%':>10}{'β@1%':>10}{'α@5%':>10}{'β@5%':>10}")
    print("-"*70)
    for cfg in configs:
        a2 = np.mean(acc[cfg]["alpha"]["tau2.0"])*100
        b2 = np.mean(acc[cfg]["beta"]["tau2.0"])*100
        a1 = np.mean(acc[cfg]["alpha"]["tau1%"])*100
        b1 = np.mean(acc[cfg]["beta"]["tau1%"])*100
        a5 = np.mean(acc[cfg]["alpha"]["tau5%"])*100
        b5 = np.mean(acc[cfg]["beta"]["tau5%"])*100
        print(f"{cfg:<20}{a2:>9.2f}%{b2:>9.2f}%{a1:>9.2f}%{b1:>9.2f}%{a5:>9.2f}%{b5:>9.2f}%")

    # IG 权重贡献
    print("\n--- IG 权重贡献（β@5% FPR，固定 IGSW 生成）---")
    b = {c: np.mean(acc[c]["beta"]["tau5%"])*100 for c in configs}
    print(f"Uniform-g -> IG-g   （IG权重贡献）: β {b['IGSW+Uniform-g']:.2f}% -> {b['IGSW+IG-g']:.2f}% (Δ{b['IGSW+IG-g']-b['IGSW+Uniform-g']:+.2f}%)")
    print(f"Uniform-g -> EWD-g  （SE权重对照）: β {b['IGSW+Uniform-g']:.2f}% -> {b['IGSW+EWD-g']:.2f}% (Δ{b['IGSW+EWD-g']-b['IGSW+Uniform-g']:+.2f}%)")

    out = {"N": N, "generator": "IGSW tanh", "configs": {}}
    for cfg in configs:
        out["configs"][cfg] = {
            "alpha": {t: float(np.mean(acc[cfg]["alpha"][t])) for t in TAUS},
            "beta": {t: float(np.mean(acc[cfg]["beta"][t])) for t in TAUS},
        }
    json.dump(out, open("results/analysis/igsw_ablation.json","w"), indent=2)
    print("\nSaved: results/analysis/igsw_ablation.json")


if __name__ == "__main__":
    main()
