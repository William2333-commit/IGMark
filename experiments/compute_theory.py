"""理论分析数值计算 v2：第一组(固定KGW生成) + 表A(经典γ) + 表B(条件g_t) + Λ + 均值±std。

对齐 EWD 协议：固定生成端 KGW(δ=2.0)，比较 KGW/SWEET/EWD/IGWD 检测器。
所有方法用相同 δ_t=2.0 和 r_t，区别仅在检测权重 a_t 和零假设概率 q_t。

用法: CUDA_VISIBLE_DEVICES=2 python experiments/compute_theory.py [--max_samples 164] [--wm_file results/humaneval/compare/kgw.jsonl]
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

# ===== 配置（核对后）=====
GAMMA = 0.5
PREFIX_LEN = 1
HASH_KEY = 15485863
DELTA_REF = 2.0
# 第一组：固定 KGW 生成
FIXED_DELTA = 2.0
# KGW
# SWEET
SWEET_EPS = 0.9  # config/SWEET.json，HumanEval 实验一致
# EWD spike entropy τ (z_value)
ALPHA_EWD = math.exp(2.0)  # e^δ
Z_VALUE = ((1 - GAMMA) * (ALPHA_EWD - 1)) / (1 - GAMMA + ALPHA_EWD * GAMMA)  # ≈0.7616
# IGSW tanh（第二组用）
D_MIN, D_MAX, K_IG, C0 = 1.0, 3.0, 12, 0.15
TAUS = {"tau2.0": 2.0, "tau1%(2.3263)": 2.3263, "tau5%(1.6449)": 1.6449}


def spike_entropy(probs, tau=Z_VALUE):
    return float((probs / (1.0 + tau * probs)).sum())


def shannon_entropy(probs):
    p = probs[probs > 0]
    return float(-(p * torch.log(p)).sum())


def ig_from_g(g, delta_ref=DELTA_REF):
    alpha = math.exp(delta_ref)
    z = 1.0 + (alpha - 1.0) * g
    q_g = (alpha * g) / max(z, 1e-12)
    return q_g * delta_ref - math.log(max(z, 1e-12))


def r_t(g, delta=FIXED_DELTA):
    a = math.exp(delta)
    return (g * a) / (1.0 + (a - 1.0) * g)


def compute_rows(tokens, probs_list, utils):
    rows = []
    for idx in range(PREFIX_LEN, len(tokens)):
        probs_t = probs_list[idx]
        if probs_t is None: continue
        greenlist = utils.get_greenlist_ids(tokens[:idx])
        g = float(probs_t[greenlist].sum())
        ig = ig_from_g(g)
        H = shannon_entropy(probs_t)
        SE = spike_entropy(probs_t)
        is_green = int(tokens[idx].item() in greenlist)
        rows.append({"g": g, "ig": ig, "H": H, "SE": SE, "is_green": is_green})
    return rows


def system_compute(rows, weight_fn, q_fn, delta=FIXED_DELTA):
    """算 μ0/σ0²/μ0_hat/σ0_hat²/μ1/σ1² 和 Λ。
    weight_fn(r)->a; q_fn(r)->q (零假设概率)。"""
    g = np.array([r["g"] for r in rows])
    a = np.array([weight_fn(r) for r in rows])
    q = np.array([q_fn(r) for r in rows])
    r = np.array([r_t(g[i], delta) for i in range(len(g))])
    mu0 = float((a * g).sum())
    sig0 = float((a*a * g*(1-g)).sum())
    mu0h = float((a * q).sum())
    sig0h = float((a*a * q*(1-q)).sum())
    mu1 = float((a * r).sum())
    sig1 = float((a*a * r*(1-r)).sum())
    return mu0, sig0, mu0h, sig0h, mu1, sig1


def classic_compute(rows, weight_fn, delta=FIXED_DELTA):
    """表A：经典 γ 零假设。q=γ, μ0=Σa*γ(假设g_t=γ), σ0=Σa²γ(1-γ)。α=1-Φ(τ)。"""
    a = np.array([weight_fn(r) for r in rows])
    g = np.array([r["g"] for r in rows])
    r = np.array([r_t(g[i], delta) for i in range(len(g))])
    # 经典：g_t=γ，所以 μ0=μ0_hat=Σa*γ, σ0=σ0_hat=Σa²γ(1-γ)
    mu0 = float((a * GAMMA).sum())
    sig0 = float((a*a * GAMMA * (1-GAMMA)).sum())
    mu1 = float((a * r).sum())
    sig1 = float((a*a * r*(1-r)).sum())
    return mu0, sig0, mu0, sig0, mu1, sig1  # μ0_hat=μ0, σ0_hat=σ0


def theory_errors(mu0, sig0, mu0h, sig0h, mu1, sig1, tau):
    c = mu0h + tau * math.sqrt(max(sig0h, 1e-12))
    s0 = math.sqrt(max(sig0, 1e-12))
    s1 = math.sqrt(max(sig1, 1e-12))
    alpha = 1 - norm.cdf((c - mu0) / s0)
    beta = norm.cdf((c - mu1) / s1)
    Lambda = (mu1 - c) / s1  # 标准化检测间隔，β=Φ(-Λ)
    return alpha, beta, Lambda


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_samples", type=int, default=164)
    ap.add_argument("--wm_file", default="results/humaneval/compare/kgw.jsonl",
                    help="第一组用 KGW 生成文本")
    args = ap.parse_args()

    print(f"配置: γ={GAMMA}, δ={FIXED_DELTA}, SWEET ε={SWEET_EPS}, EWD z_value={Z_VALUE:.4f}")
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

    systems = ["KGW", "SWEET", "EWD", "IGWD"]
    # 第一组：δ=2.0 固定，权重不同，q 不同
    # 条件表B: q=γ(KGW/SWEET/EWD), q=g_t(IGWD)
    cond = {n: {"mu0":[], "sig0":[], "mu0h":[], "sig0h":[], "mu1":[], "sig1":[], "Lambda":[]} for n in systems}
    # 经典表A: q=γ, μ0=Σa*γ
    classic = {n: {"mu0":[], "sig0":[], "mu1":[], "sig1":[], "Lambda":[]} for n in systems}

    for i, d in enumerate(data):
        text = d.get("watermarked_text", d.get("watermarked", ""))
        if not text.strip(): continue
        encoded = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(model.device)
        if len(encoded) <= PREFIX_LEN: continue
        with torch.no_grad():
            probs_list = utils.calculate_probabilities(model, encoded)
        rows = compute_rows(encoded, probs_list, utils)
        if not rows: continue

        se_min = min(r["SE"] for r in rows)
        ig_min = min(r["ig"] for r in rows)
        # 权重函数
        w_fns = {
            "KGW": lambda r: 1.0,
            "SWEET": lambda r: 1.0 if r["H"] > SWEET_EPS else 0.0,
            "EWD": lambda r: max(r["SE"] - se_min, 0.0),
            "IGWD": lambda r: max(r["ig"] - ig_min, 0.0),
        }
        # 条件 q
        q_fns = {
            "KGW": lambda r: GAMMA,
            "SWEET": lambda r: GAMMA,
            "EWD": lambda r: GAMMA,
            "IGWD": lambda r: r["g"],
        }
        for n in systems:
            # 条件(表B)
            mu0, sig0, mu0h, sig0h, mu1, sig1 = system_compute(rows, w_fns[n], q_fns[n])
            for k, v in zip(["mu0","sig0","mu0h","sig0h","mu1","sig1"], [mu0,sig0,mu0h,sig0h,mu1,sig1]):
                cond[n][k].append(v)
            # 经典(表A)
            m0, s0, m0h, s0h, m1, s1 = classic_compute(rows, w_fns[n])
            for k, v in zip(["mu0","sig0","mu1","sig1"], [m0,s0,m1,s1]):
                classic[n][k].append(v)
        if (i+1) % 20 == 0: print(f"  {i+1}/{len(data)}", flush=True)

    N = len(cond["KGW"]["mu0"])
    print(f"\n有效样本: {N}\n")

    def mean_std(x): return f"{np.mean(x):.3f}±{np.std(x):.3f}"

    for table_name, tbl, use_classic in [("表A(经典γ零假设)", classic, True), ("表B(条件g_t零假设)", cond, False)]:
        print(f"===== {table_name} =====")
        for tau_name, tau in TAUS.items():
            print(f"\n--- {tau_name} (τ={tau}) ---")
            print(f"{'方法':<8}{'μ0':>16}{'σ0²':>16}{'μ1':>16}{'σ1²':>16}{'α(%)':>10}{'β(%)':>10}{'Λ':>8}")
            for n in systems:
                if use_classic:
                    mu0, sig0, mu1, sig1 = tbl[n]["mu0"], tbl[n]["sig0"], tbl[n]["mu1"], tbl[n]["sig1"]
                    alphas, betas, lambdas = [], [], []
                    for i in range(N):
                        a, b, L = theory_errors(mu0[i], sig0[i], mu0[i], sig0[i], mu1[i], sig1[i], tau)
                        alphas.append(a); betas.append(b); lambdas.append(L)
                    print(f"{n:<8}{mean_std(mu0):>16}{mean_std(sig0):>16}{mean_std(mu1):>16}{mean_std(sig1):>16}"
                          f"{np.mean(alphas)*100:>9.2f}%{np.mean(betas)*100:>9.2f}%{np.mean(lambdas):>8.3f}")
                else:
                    mu0, sig0, mu0h, sig0h, mu1, sig1 = (tbl[n][k] for k in ["mu0","sig0","mu0h","sig0h","mu1","sig1"])
                    alphas, betas, lambdas = [], [], []
                    for i in range(N):
                        a, b, L = theory_errors(mu0[i], sig0[i], mu0h[i], sig0h[i], mu1[i], sig1[i], tau)
                        alphas.append(a); betas.append(b); lambdas.append(L)
                    print(f"{n:<8}{mean_std(mu0):>16}{mean_std(sig0):>16}{mean_std(mu1):>16}{mean_std(sig1):>16}"
                          f"{np.mean(alphas)*100:>9.2f}%{np.mean(betas)*100:>9.2f}%{np.mean(lambdas):>8.3f}")
            print()

    # 保存
    out = {"N": N, "config": {"γ":GAMMA,"δ":FIXED_DELTA,"SWEET_ε":SWEET_EPS,"EWD_z_value":Z_VALUE},
           "classic": {n: {k: [float(x) for x in v] for k,v in tbl[n].items()} for n in systems},
           "conditional": {n: {k: [float(x) for x in v] for k,v in tbl[n].items()} for n in systems}}
    json.dump(out, open("results/analysis/theory_errors_v2.json","w"), indent=2)
    print("Saved: results/analysis/theory_errors_v2.json")


if __name__ == "__main__":
    main()
