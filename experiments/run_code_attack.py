"""代码专属攻击鲁棒性实验：AST 变量重命名 + Qwen refactor。

两阶段：
  apply  : 施加攻击 + 功能测试（CPU，不需 GPU），保存 attacked jsonl + 功能保持率
  detect : 加载 StarCoder + 水印检测器，对 attacked 文本检测，算 AUROC/TPR/F1

用法:
  # 攻击1 apply（CPU，立即跑）
  python experiments/run_code_attack.py --phase apply --attack rename --dataset HumanEval
  python experiments/run_code_attack.py --phase apply --attack rename --dataset MBPP

  # 攻击2 apply（需 Qwen，GPU3）
  CUDA_VISIBLE_DEVICES=3 python experiments/run_code_attack.py --phase apply --attack qwen --dataset HumanEval

  # detect（需 StarCoder，等 GPU 空闲）
  CUDA_VISIBLE_DEVICES=X python experiments/run_code_attack.py --phase detect --dataset HumanEval
"""
import os, sys, json, argparse, tempfile, threading
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.dataset import HumanEvalDataset
from evaluation.mbpp_dataset import MBPPDataset
from evaluation.tools.text_editor import TruncateTaskTextEditor, CodeGenerationTextEditor
from evaluation.tools.text_quality_analyzer import PassOrNotJudger

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PID = os.getpid()
HASH = 15485863
IGSW_CFG = {"algorithm_name": "IGSW", "gamma": 0.5, "delta_ref": 2.0, "hash_key": HASH,
    "z_threshold": 4.0, "prefix_length": 1, "eps": 1e-12, "visualize_mode": "raw_ig",
    "function": "tanh", "delta_min": 1.0, "delta_max": 3.0, "c0": 0.15, "k": 12}
SWEET_CFG = {"algorithm_name": "SWEET", "gamma": 0.5, "delta": 2.0, "hash_key": HASH,
    "z_threshold": 4.0, "prefix_length": 1, "entropy_threshold": 0.65}

ALGS = ["IGSW", "KGW", "SWEET", "DBW"]
ALG_FILE_PREFIX = {"IGSW": "igsw", "KGW": "kgw", "SWEET": "sweet", "DBW": "dbw"}

EXEC_TIMEOUT = 5
class TimeoutJudger(PassOrNotJudger):
    def analyze(self, text, reference):
        result = [0]
        def _run(): result[0] = PassOrNotJudger.analyze(self, text, reference)
        t = threading.Thread(target=_run, daemon=True); t.start(); t.join(timeout=EXEC_TIMEOUT)
        return result[0]


def get_judger(ds): return TimeoutJudger() if ds == "MBPP" else PassOrNotJudger()


def load_dataset(ds, max_samples):
    if ds == "HumanEval":
        return HumanEvalDataset("dataset/human_eval/test.jsonl", max_samples=max_samples)
    return MBPPDataset(split="test", max_samples=max_samples, use_few_shot=True)


def load_wm_texts(ds, alg, n):
    """从现有水印 jsonl 读 watermarked_text + unwatermarked_text + prompt。"""
    folder = {"HumanEval": "results/humaneval/compare", "MBPP": "results/mbpp/compare"}[ds]
    prefix = ALG_FILE_PREFIX[alg]
    # MBPP SWEET 文件名特殊
    if ds == "MBPP" and alg == "SWEET":
        prefix = "sweet_065"
    if ds == "MBPP" and alg == "IGSW":
        prefix = "igsw_k=12"
    if ds == "HumanEval" and alg == "IGSW":
        prefix = "igsw_k12"
    fname = next((f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(".jsonl")), None)
    if not fname:
        print(f"  [WARN] {alg}: jsonl not found in {folder}")
        return None
    with open(f"{folder}/{fname}") as f:
        data = [json.loads(l) for l in f]
    wm = [d.get("watermarked_text", d.get("watermarked", "")) for d in data][:n]
    non = [d.get("unwatermarked_text", "") for d in data][:n]
    prompts = [d.get("prompt", "") for d in data][:n]
    return wm, non, prompts


# ============ apply 阶段 ============
def phase_apply(args):
    from experiments.attacks.variable_rename import VariableRenameEditor
    ds_dir = f"results/code_attack/{args.dataset.lower()}/{args.attack}"
    os.makedirs(ds_dir, exist_ok=True)

    n = args.max_samples if args.max_samples > 0 else (164 if args.dataset == "HumanEval" else 500)
    dataset = load_dataset(args.dataset, n)
    N = dataset.prompt_nums
    judger = get_judger(args.dataset)
    trunc, code_editor = TruncateTaskTextEditor(), CodeGenerationTextEditor()

    # 攻击 editor
    if args.attack == "rename":
        def make_editor(seed): return VariableRenameEditor(depth=args.depth, seed=seed)
    elif args.attack == "qwen":
        from experiments.attacks.qwen_refactor import QwenRefactorEditor
        qwen = QwenRefactorEditor(device="cuda")
        def make_editor(seed): return qwen
    else:
        raise ValueError(args.attack)

    seeds = list(range(args.seeds))
    all_keep = {}
    for alg in ALGS:
        loaded = load_wm_texts(args.dataset, alg, N)
        if loaded is None: continue
        wm_texts, non_texts, prompts = loaded
        print(f"\n{'='*50}\n{alg} ({len(wm_texts)} texts)\n{'='*50}")
        all_keep[alg] = {}
        for seed in seeds:
            editor = make_editor(seed)
            attacked, kept = [], 0
            for i, body in enumerate(tqdm(wm_texts, desc=f"{alg}/seed{seed}")):
                ref = dataset.get_reference(i)
                # 用 reference['task']（单函数签名，纯代码）而非 prompt（可能含 few-shot 自然语言）
                task = ref['task'].replace('\r\n', '\n').replace('\t', '    ').rstrip()
                body_clean = body.replace('\r\n', '\n').replace('\t', '    ')
                full = task + '\n' + body_clean
                att_full = editor.edit(full)
                if att_full.startswith(task):
                    att_body = att_full[len(task):].lstrip('\n')
                else:
                    att_body = body_clean  # fallback
                attacked.append(att_body)
                kept += judger.analyze(att_body, ref)
            keep_rate = kept / max(len(attacked), 1)
            all_keep[alg][seed] = {"keep_rate": round(keep_rate, 4), "kept": kept, "total": len(attacked)}
            print(f"  seed{seed}: functional keep = {keep_rate:.4f} ({kept}/{len(attacked)})")
            # 保存 attacked jsonl
            out = f"{ds_dir}/{alg.lower()}_seed{seed}.jsonl"
            with open(out, "w", encoding="utf-8") as f:
                for i, a in enumerate(attacked):
                    f.write(json.dumps({"idx": i, "prompt": prompts[i], "watermarked_text": wm_texts[i],
                                        "attacked_text": a, "unwatermarked_text": non_texts[i]},
                                       ensure_ascii=False) + "\n")
            print(f"  saved: {out}")

    with open(f"{ds_dir}/functional_keep.json", "w") as f:
        json.dump(all_keep, f, ensure_ascii=False, indent=2)
    print(f"\nFunctional keep saved: {ds_dir}/functional_keep.json")


# ============ detect 阶段 ============
def compute_metrics(wm_scores, non_scores):
    if len(wm_scores) < 2 or len(non_scores) < 2:
        return {"AUROC": 0, "TPR@1%": 0, "F1@1%": 0, "TPR@5%": 0, "F1@5%": 0, "Best-F1": 0, "pairs": 0}
    yt = np.array([1]*len(wm_scores) + [0]*len(non_scores))
    ys = np.array(wm_scores + non_scores)
    auroc = float(roc_auc_score(yt, ys))
    fpr, tpr, ths = roc_curve(yt, ys)
    def m_at(t):
        valid = np.where(fpr <= t)[0]
        if len(valid) == 0: return 0.0, 0.0
        idx = valid[np.argmax(tpr[valid])]
        tp = int(np.sum(np.array(wm_scores) >= ths[idx])); fp = int(np.sum(np.array(non_scores) >= ths[idx]))
        fn = len(wm_scores) - tp; prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
        return float(rec), float(2*prec*rec/max(prec+rec,1e-12))
    tpr1, f1_1 = m_at(0.01); tpr5, f1_5 = m_at(0.05)
    best_f1 = 0.0
    for th in sorted(set(ys), reverse=True):
        tp = int(np.sum(np.array(wm_scores) >= th)); fp = int(np.sum(np.array(non_scores) >= th))
        fn = len(wm_scores) - tp; prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
        f1 = 2*prec*rec/max(prec+rec,1e-12)
        if f1 > best_f1: best_f1 = f1
    return {"AUROC": round(auroc,4), "TPR@1%": round(tpr1,4), "F1@1%": round(f1_1,4),
            "TPR@5%": round(tpr5,4), "F1@5%": round(f1_5,4), "Best-F1": round(best_f1,4),
            "pairs": len(wm_scores)}


def parse_score(r):
    if isinstance(r, dict):
        for k in ["score", "z_score", "z"]:
            if k in r: return float(r[k])
    if isinstance(r, (int, float)): return float(r)
    return None


def safe_detect(wm, text):
    if not text or not text.strip(): return None
    try:
        s = parse_score(wm.detect_watermark(text, return_dict=True))
        import math
        if s is None or s != s or math.isinf(s): return None
        return float(s)
    except: return None


def load_wm_instance(alg, tcfg):
    if alg == "IGSW":
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(IGSW_CFG, tmp); tmp.close()
        wm = AutoWatermark.load("IGSW", algorithm_config=tmp.name, transformers_config=tcfg)
        os.unlink(tmp.name)
    elif alg == "SWEET":
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(SWEET_CFG, tmp); tmp.close()
        wm = AutoWatermark.load("SWEET", algorithm_config=tmp.name, transformers_config=tcfg)
        os.unlink(tmp.name)
    else:
        wm = AutoWatermark.load(alg, algorithm_config=f"config/{alg}.json", transformers_config=tcfg)
    return wm


def phase_detect(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    MODEL = "bigcode/starcoder"
    print("Loading StarCoder...")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cuda:0", torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()
    tcfg = TransformersConfig(model=model, tokenizer=tok, vocab_size=len(tok), device="cuda",
        max_new_tokens=512, do_sample=True, temperature=0.2, top_p=0.95)

    wms = {alg: load_wm_instance(alg, tcfg) for alg in ALGS}
    attacks = ["rename"]  # 只检测 rename（随机词版）；qwen 无效跳过
    seeds = list(range(args.seeds))
    base_dir = f"results/code_attack/{args.dataset.lower()}"

    all_results = []
    for alg in ALGS:
        # 基线 None: 用原始 watermarked_text 检测
        loaded = load_wm_texts(args.dataset, alg, 9999)
        if loaded is None: continue
        wm_texts, non_texts, _ = loaded
        wm_s = [s for t in wm_texts if (s := safe_detect(wms[alg], t)) is not None]
        non_s = [s for t in non_texts if (s := safe_detect(wms[alg], t)) is not None]
        n = min(len(wm_s), len(non_s))
        m = compute_metrics(wm_s[:n], non_s[:n])
        m.update({"algorithm": alg, "attack": "None", "seed": 0})
        print(f"  {alg}/None: AUROC={m['AUROC']} TPR@5%={m['TPR@5%']} Best-F1={m['Best-F1']}")
        all_results.append(m)

        for attack in attacks:
            for seed in seeds:
                f = f"{base_dir}/{attack}/{alg.lower()}_seed{seed}.jsonl"
                if not os.path.exists(f):
                    print(f"  [skip] {f} not found"); continue
                with open(f) as fh: data = [json.loads(l) for l in fh]
                att = [d["attacked_text"] for d in data]
                non = [d.get("unwatermarked_text","") for d in data]
                wm_s = [s for t in tqdm(att, desc=f"{alg}/{attack}/s{seed} detect") if (s := safe_detect(wms[alg], t)) is not None]
                non_s = [s for t in non if (s := safe_detect(wms[alg], t)) is not None]
                n = min(len(wm_s), len(non_s))
                m = compute_metrics(wm_s[:n], non_s[:n])
                m.update({"algorithm": alg, "attack": attack, "seed": seed})
                print(f"  {alg}/{attack}/s{seed}: AUROC={m['AUROC']} TPR@5%={m['TPR@5%']} Best-F1={m['Best-F1']} pairs={m['pairs']}")
                all_results.append(m)

    out = {"dataset": args.dataset, "model": MODEL, "results": all_results}
    with open(f"{base_dir}/detection.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {base_dir}/detection.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["apply", "detect"])
    ap.add_argument("--attack", default="rename", choices=["rename", "qwen"])
    ap.add_argument("--dataset", default="HumanEval", choices=["HumanEval", "MBPP"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()
    if args.phase == "apply": phase_apply(args)
    else: phase_detect(args)


if __name__ == "__main__":
    main()
