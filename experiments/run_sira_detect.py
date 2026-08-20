"""SIRA 攻击后检测：对 SIRA attack.json 的 attack_text 检测水印 + PPL。

WT2 用 OPT-1.3B 生成水印，检测也用 OPT-1.3B。
对比：None（原 watermarked_text）vs SIRA attack_text。

用法: CUDA_VISIBLE_DEVICES=X python experiments/run_sira_detect.py
"""
import os, sys, json, math, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch, numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModelForCausalLM
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL = "facebook/opt-1.3b"
HASH = 15485863
ATTACK_DIR = "Self-information-Rewrite-Attack-main/Self-information-Rewrite-Attack-main/dataset/wt2/attack"
ALGS = ["IGSW", "KGW", "SWEET", "DBW"]
ALG_CFG = {
    "IGSW": {"algorithm_name":"IGSW","gamma":0.5,"delta_ref":2.0,"hash_key":HASH,"z_threshold":4.0,"prefix_length":1,"eps":1e-12,"visualize_mode":"raw_ig","function":"tanh","delta_min":1.0,"delta_max":3.0,"c0":0.15,"k":12},
    "KGW": None, "SWEET": None, "DBW": None,
}


def compute_metrics(wm_s, non_s):
    if len(wm_s) < 2 or len(non_s) < 2: return {"AUROC":0,"TPR@1%":0,"F1@1%":0,"TPR@5%":0,"F1@5%":0,"Best-F1":0}
    yt = np.array([1]*len(wm_s)+[0]*len(non_s)); ys = np.array(wm_s+non_s)
    auroc = float(roc_auc_score(yt, ys))
    fpr, tpr, ths = roc_curve(yt, ys)
    def m_at(t):
        v = np.where(fpr<=t)[0]
        if len(v)==0: return 0,0
        i = v[np.argmax(tpr[v])]
        tp=int(np.sum(np.array(wm_s)>=ths[i])); fp=int(np.sum(np.array(non_s)>=ths[i])); fn=len(wm_s)-tp
        p=tp/max(tp+fp,1); r=tp/max(tp+fn,1); return float(r), float(2*p*r/max(p+r,1e-12))
    t1,f1=m_at(0.01); t5,f5=m_at(0.05)
    bf=0
    for th in sorted(set(ys),reverse=True):
        tp=int(np.sum(np.array(wm_s)>=th)); fp=int(np.sum(np.array(non_s)>=th)); fn=len(wm_s)-tp
        p=tp/max(tp+fp,1); r=tp/max(tp+fn,1); f=2*p*r/max(p+r,1e-12)
        if f>bf: bf=f
    return {"AUROC":round(auroc,4),"TPR@1%":round(t1,4),"F1@1%":round(f1,4),"TPR@5%":round(t5,4),"F1@5%":round(f5,4),"Best-F1":round(bf,4)}


def parse_score(r):
    if isinstance(r,dict):
        for k in ["score","z_score","z"]:
            if k in r: return float(r[k])
    if isinstance(r,(int,float)): return float(r)
    return None


def safe_detect(wm, text):
    if not text or not text.strip(): return None
    try:
        s = parse_score(wm.detect_watermark(text, return_dict=True))
        if s is None or s!=s or math.isinf(s): return None
        return float(s)
    except: return None


def load_wm(alg, tcfg):
    if alg == "IGSW":
        fd, p = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd,'w') as f: json.dump(ALG_CFG["IGSW"], f)
        wm = AutoWatermark.load("IGSW", algorithm_config=p, transformers_config=tcfg); os.unlink(p)
    else:
        wm = AutoWatermark.load(alg, algorithm_config=f"config/{alg}.json", transformers_config=tcfg)
    return wm


def clean_attack_text(text):
    """SIRA attack_text 可能含前缀 'Here is the complete paragraph:\n\n'，去掉。"""
    prefixes = ["Here is the complete paragraph:", "Here is the complete paragraph:\n", "Here is a complete paragraph:"]
    for pre in prefixes:
        if text.startswith(pre):
            text = text[len(pre):].lstrip()
            break
    return text.strip()


def main():
    print(f"Loading {MODEL}...")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cuda:0", torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()
    tcfg = TransformersConfig(model=model, tokenizer=tok, vocab_size=len(tok), device="cuda",
                              max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.95)

    wms = {alg: load_wm(alg, tcfg) for alg in ALGS}
    all_results = []

    for alg in ALGS:
        attack_file = f"{ATTACK_DIR}/{alg}_attack.json"
        if not os.path.exists(attack_file):
            print(f"[skip] {attack_file} not found"); continue
        data = [json.loads(l) for l in open(attack_file)]
        wm = wms[alg]

        # None 基线：原 watermarked_text
        none_wm = [s for t in tqdm(data, desc=f"{alg}/None detect") if (s:=safe_detect(wm, t.get("watermarked_text",""))) is not None]
        none_non = [s for t in tqdm(data, desc=f"{alg}/None non") if (s:=safe_detect(wm, t.get("unwatermarked_text",""))) is not None]
        n = min(len(none_wm), len(none_non))
        m_none = compute_metrics(none_wm[:n], none_non[:n])
        m_none.update({"algorithm":alg, "attack":"None"})
        print(f"  {alg}/None: AUROC={m_none['AUROC']} TPR@5%={m_none['TPR@5%']} Best-F1={m_none['Best-F1']}")
        all_results.append(m_none)

        # SIRA attack_text
        att_wm = [s for t in tqdm(data, desc=f"{alg}/SIRA detect") if (s:=safe_detect(wm, clean_attack_text(t.get("attack_text","")))) is not None]
        att_non = [s for t in tqdm(data, desc=f"{alg}/SIRA non") if (s:=safe_detect(wm, t.get("unwatermarked_text",""))) is not None]
        n = min(len(att_wm), len(att_non))
        m_att = compute_metrics(att_wm[:n], att_non[:n])
        m_att.update({"algorithm":alg, "attack":"SIRA"})
        print(f"  {alg}/SIRA: AUROC={m_att['AUROC']} TPR@5%={m_att['TPR@5%']} Best-F1={m_att['Best-F1']}")
        all_results.append(m_att)

    # PPL (用 OPT-1.3B 算 PPL)
    print("\n=== PPL ===")
    for alg in ALGS:
        attack_file = f"{ATTACK_DIR}/{alg}_attack.json"
        if not os.path.exists(attack_file): continue
        data = [json.loads(l) for l in open(attack_file)]
        ppls = []
        for d in tqdm(data[:50], desc=f"{alg} PPL"):
            text = clean_attack_text(d.get("attack_text",""))
            if not text.strip(): continue
            try:
                enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
                with torch.no_grad():
                    out = model(**enc, labels=enc["input_ids"])
                ppls.append(float(torch.exp(out.loss)))
            except: pass
        print(f"  {alg} SIRA PPL: {np.mean(ppls):.2f} (n={len(ppls)})")
        all_results.append({"algorithm":alg, "attack":"SIRA_PPL", "PPL": round(float(np.mean(ppls)),2)})

    out = {"dataset":"WikiText-2", "model":"OPT-1.3B", "attack_model":"Llama-3-8B-Instruct", "results":all_results}
    os.makedirs("results/sira", exist_ok=True)
    json.dump(out, open("results/sira/sira_detection.json","w"), ensure_ascii=False, indent=2)
    print("\n=== 汇总 ===")
    print(f"{'方法':<8}{'None AUROC':>12}{'SIRA AUROC':>12}{'Δ':>10}{'None TPR@5%':>12}{'SIRA TPR@5%':>12}")
    print("-"*66)
    for alg in ALGS:
        none = next((r for r in all_results if r.get("algorithm")==alg and r.get("attack")=="None"), {})
        sira = next((r for r in all_results if r.get("algorithm")==alg and r.get("attack")=="SIRA"), {})
        na = none.get("AUROC",0); sa = sira.get("AUROC",0)
        nt = none.get("TPR@5%",0); st = sira.get("TPR@5%",0)
        print(f"{alg:<8}{na:>12}{sa:>12}{sa-na:>+10}{nt:>12}{st:>12}")
    print(f"\nSaved: results/sira/sira_detection.json")


if __name__ == "__main__":
    main()
