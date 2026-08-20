"""SIRA 代码攻击检测：对 HumanEval/MBPP 的 SIRA attack.json 检测水印 + PASS@1。

用 StarCoder 检测，SubprocessJudger 跑 PASS@1。

用法: CUDA_VISIBLE_DEVICES=X python experiments/run_sira_code_detect.py --dataset HumanEval
"""
import os, sys, json, math, tempfile, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch, numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModelForCausalLM
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.dataset import HumanEvalDataset
from evaluation.mbpp_dataset import MBPPDataset
from evaluation.tools.text_quality_analyzer import PassOrNotJudger

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL = "bigcode/starcoder"
HASH = 15485863
SIRA_BASE = "Self-information-Rewrite-Attack-main/Self-information-Rewrite-Attack-main"
ALGS = ["IGSW", "KGW", "SWEET", "DBW"]
IGSW_CFG = {"algorithm_name":"IGSW","gamma":0.5,"delta_ref":2.0,"hash_key":HASH,"z_threshold":4.0,"prefix_length":1,"eps":1e-12,"visualize_mode":"raw_ig","function":"tanh","delta_min":1.0,"delta_max":3.0,"c0":0.15,"k":12}
EXEC_TIMEOUT = 5

class SubprocessJudger(PassOrNotJudger):
    def analyze(self, text, reference):
        check_program = (reference['task'] + '\n' + text + '\n' + reference['test'] + '\n' + f"check({reference['entry_point']})")
        import subprocess, tempfile
        fd, path = tempfile.mkstemp(suffix='.py')
        try:
            with os.fdopen(fd, 'w') as f: f.write(check_program)
            try:
                r = subprocess.run(['python3', path], timeout=EXEC_TIMEOUT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return 1 if r.returncode == 0 else 0
            except: return 0
        finally:
            try: os.unlink(path)
            except: pass

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

def clean_attack_text(text):
    for pre in ["Here is the complete paragraph:\n\n", "Here is the complete paragraph:\n", "Here is the complete paragraph:", "Here is a complete paragraph:\n\n"]:
        if text.startswith(pre):
            text = text[len(pre):]
            break
    return text.strip()

def load_wm(alg, tcfg):
    if alg == "IGSW":
        fd, p = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd,'w') as f: json.dump(IGSW_CFG, f)
        wm = AutoWatermark.load("IGSW", algorithm_config=p, transformers_config=tcfg); os.unlink(p)
    else:
        wm = AutoWatermark.load(alg, algorithm_config=f"config/{alg}.json", transformers_config=tcfg)
    return wm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HumanEval", choices=["HumanEval","MBPP"])
    args = ap.parse_args()
    ds = args.dataset.lower()

    print(f"Loading {MODEL}...")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cuda:0", torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()
    tcfg = TransformersConfig(model=model, tokenizer=tok, vocab_size=len(tok), device="cuda", max_new_tokens=512, do_sample=True, temperature=0.2, top_p=0.95)

    wms = {alg: load_wm(alg, tcfg) for alg in ALGS}
    # 加载数据集（PASS@1 用）
    if args.dataset == "HumanEval":
        dataset = HumanEvalDataset("dataset/human_eval/test.jsonl", max_samples=200)
    else:
        dataset = MBPPDataset(split="test", max_samples=600, use_few_shot=True)
    judger = SubprocessJudger()

    attack_dir = f"{SIRA_BASE}/dataset/{ds}/attack"
    all_results = []

    for alg in ALGS:
        f = f"{attack_dir}/{alg}_attack.json"
        if not os.path.exists(f):
            print(f"[skip] {f}"); continue
        data = [json.loads(l) for l in open(f)]
        wm = wms[alg]

        # None 基线
        none_wm = [s for t in data if (s:=safe_detect(wm, t.get("watermarked_text",""))) is not None]
        none_non = [s for t in data if (s:=safe_detect(wm, t.get("unwatermarked_text",""))) is not None]
        n = min(len(none_wm), len(none_non))
        m_none = compute_metrics(none_wm[:n], none_non[:n])
        m_none.update({"algorithm":alg, "attack":"None"})
        print(f"  {alg}/None: AUROC={m_none['AUROC']} TPR@5%={m_none['TPR@5%']}")

        # SIRA attack_text
        att_wm = [s for t in data if (s:=safe_detect(wm, clean_attack_text(t.get("attack_text","")))) is not None]
        att_non = [s for t in data if (s:=safe_detect(wm, t.get("unwatermarked_text",""))) is not None]
        n2 = min(len(att_wm), len(att_non))
        m_att = compute_metrics(att_wm[:n2], att_non[:n2])
        m_att.update({"algorithm":alg, "attack":"SIRA"})
        print(f"  {alg}/SIRA: AUROC={m_att['AUROC']} TPR@5%={m_att['TPR@5%']}")
        all_results.extend([m_none, m_att])

        # PASS@1
        pass_none, pass_att = 0, 0
        for d in tqdm(data, desc=f"{alg} PASS@1"):
            idx = d.get("idx", 0)
            try: ref = dataset.get_reference(idx)
            except: continue
            if not ref.get('test'): continue
            pass_none += judger.analyze(d.get("watermarked_text",""), ref)
            pass_att += judger.analyze(clean_attack_text(d.get("attack_text","")), ref)
        p_none = pass_none / max(len(data), 1)
        p_att = pass_att / max(len(data), 1)
        print(f"  {alg} PASS@1: None={p_none:.4f} SIRA={p_att:.4f}")
        all_results.append({"algorithm":alg, "attack":"SIRA_PASS@1", "pass_none":round(p_none,4), "pass_sira":round(p_att,4)})

    out = {"dataset":args.dataset, "model":MODEL, "attack_model":"Llama-3-8B-Instruct", "results":all_results}
    os.makedirs("results/sira", exist_ok=True)
    json.dump(out, open(f"results/sira/sira_{ds}_code_detection.json","w"), ensure_ascii=False, indent=2)
    print(f"\n=== {args.dataset} 汇总 ===")
    print(f"{'方法':<8}{'None AUROC':>12}{'SIRA AUROC':>12}{'Δ':>10}{'None PASS@1':>14}{'SIRA PASS@1':>14}")
    print("-"*70)
    for alg in ALGS:
        none = next((r for r in all_results if r.get("algorithm")==alg and r.get("attack")=="None"), {})
        sira = next((r for r in all_results if r.get("algorithm")==alg and r.get("attack")=="SIRA"), {})
        pa = next((r for r in all_results if r.get("algorithm")==alg and r.get("attack")=="SIRA_PASS@1"), {})
        print(f"{alg:<8}{none.get('AUROC',0):>12}{sira.get('AUROC',0):>12}{sira.get('AUROC',0)-none.get('AUROC',0):>+10}{pa.get('pass_none',0):>14}{pa.get('pass_sira',0):>14}")
    print(f"\nSaved: results/sira/sira_{ds}_code_detection.json")

if __name__ == "__main__":
    main()
