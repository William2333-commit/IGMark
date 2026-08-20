"""MBPP 对比实验：IGSW(k=12) vs KGW vs SWEET vs DBW vs IGWD

- 100 样本小份量
- 共享 unwatermarked 基线（生成一次，所有方法共用）
- 输出: results/mbpp/compare/
"""
import os, sys, json, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModelForCausalLM

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.mbpp_dataset import MBPPDataset
from evaluation.tools.text_editor import TruncateTaskTextEditor, CodeGenerationTextEditor
from evaluation.pipelines.quality_analysis import (
    ReferencedTextQualityAnalysisPipeline, QualityPipelineReturnType)
from evaluation.tools.text_quality_analyzer import PassOrNotJudger
import threading

EXEC_TIMEOUT = 5  # seconds per test

class TimeoutJudger(PassOrNotJudger):
    """PassOrNotJudger with timeout to prevent infinite-loop hangs."""
    def analyze(self, text, reference):
        result = [0]
        def _run():
            result[0] = PassOrNotJudger.analyze(self, text, reference)
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=EXEC_TIMEOUT)
        return result[0]

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PID = os.getpid()
LOG_DIR = "results/mbpp/compare"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = f"{LOG_DIR}/run_{PID}.log"

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
print(f"[PID={PID}] MBPP compare: IGSW k=12 vs KGW vs SWEET vs DBW vs IGWD")

MODEL_NAME = "bigcode/starcoder"
MAX_SAMPLES = 50
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2
TOP_P = 0.95

def extract_pass(obj):
    if isinstance(obj, (int, float)): return float(obj)
    if isinstance(obj, dict):
        for v in obj.values():
            r = extract_pass(v)
            if r is not None: return r
    return None

def parse_score(result):
    if isinstance(result, dict):
        for k in ["score", "z_score", "z"]:
            if k in result: return float(result[k])
        for v in result.values():
            try: return parse_score(v)
            except: pass
    if isinstance(result, (int, float)): return float(result)
    raise ValueError(f"Cannot parse: {result}")

def safe_detect(wm, text):
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return False, None
    try:
        return True, parse_score(wm.detect_watermark(text, return_dict=True))
    except:
        return False, None

def compute_metrics(wm_scores, non_scores):
    if len(wm_scores) == 0:
        return {"AUROC": 0, "TPR@1%": 0, "F1@1%": 0, "TPR@5%": 0, "F1@5%": 0,
                "Best-F1": 0, "D": 0, "z_mean": 0, "pairs": 0}
    yt = np.array([1]*len(wm_scores) + [0]*len(non_scores))
    ys = np.array(wm_scores + non_scores)
    auroc = float(roc_auc_score(yt, ys))
    fpr, tpr, ths = roc_curve(yt, ys)
    def m_at(t):
        valid = np.where(fpr <= t)[0]
        if len(valid) == 0: return 0.0, 0.0
        idx = valid[np.argmax(tpr[valid])]
        tp = int(np.sum(np.array(wm_scores) >= ths[idx]))
        fp = int(np.sum(np.array(non_scores) >= ths[idx]))
        fn = len(wm_scores) - tp
        prec = tp / max(tp+fp, 1); rec = tp / max(tp+fn, 1)
        return float(rec), float(2*prec*rec/max(prec+rec, 1e-12))
    tpr1, f1_1 = m_at(0.01); tpr5, f1_5 = m_at(0.05)
    best_f1 = 0.0
    for th in sorted(set(ys), reverse=True):
        tp = int(np.sum(np.array(wm_scores) >= th))
        fp = int(np.sum(np.array(non_scores) >= th))
        fn = len(wm_scores) - tp
        prec = tp / max(tp+fp, 1); rec = tp / max(tp+fn, 1)
        f1 = 2*prec*rec / max(prec+rec, 1e-12)
        if f1 > best_f1: best_f1 = f1
    return {"AUROC": round(auroc, 4), "TPR@1%": round(tpr1, 4), "F1@1%": round(f1_1, 4),
            "TPR@5%": round(tpr5, 4), "F1@5%": round(f1_5, 4),
            "Best-F1": round(best_f1, 4),
            "D": round(float((auroc+tpr5)/2), 4),
            "z_mean": round(float(np.mean(wm_scores)), 4),
            "pairs": len(wm_scores)}

# ============================================================
# Load model
# ============================================================
print("Loading StarCoder...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="cuda:0", torch_dtype=torch.float16, trust_remote_code=True)
model.eval()
tcfg = TransformersConfig(model=model, tokenizer=tokenizer, vocab_size=len(tokenizer),
    device="cuda", max_new_tokens=MAX_NEW_TOKENS, do_sample=True, top_p=TOP_P, temperature=TEMPERATURE)
dataset = MBPPDataset(split="test", max_samples=MAX_SAMPLES, use_few_shot=True)
N = dataset.prompt_nums
print(f"MBPP: {N} samples")

trunc_editor = TruncateTaskTextEditor()
code_editor = CodeGenerationTextEditor()

# ============================================================
# Step 1: 共享 unwatermarked 基线（生成一次，全部共用）
# ============================================================
print("\n=== Shared unwatermarked baseline ===")
wm_base = AutoWatermark.load("IGSW", algorithm_config="config/IGSW.json", transformers_config=tcfg)
non_texts = []
for i in tqdm(range(N), desc="unwatermarked gen"):
    prompt = dataset.get_prompt(i)
    non_texts.append(code_editor.edit(
        trunc_editor.edit(wm_base.generate_unwatermarked_text(prompt), prompt), prompt))

# PASS@1 baseline pipeline
pipeline = ReferencedTextQualityAnalysisPipeline(
    dataset=dataset,
    watermarked_text_editor_list=[TruncateTaskTextEditor(), CodeGenerationTextEditor()],
    unwatermarked_text_editor_list=[TruncateTaskTextEditor(), CodeGenerationTextEditor()],
    analyzers=[TimeoutJudger()],
    unwatermarked_text_source="generated",
    show_progress=True, return_type=QualityPipelineReturnType.MEAN_SCORES)
baseline_raw = pipeline.evaluate(wm_base)
pass_non_baseline = extract_pass(baseline_raw.get("unwatermarked", {}))
print(f"Shared PASS@1_non = {pass_non_baseline:.4f}")

# ============================================================
# Step 2: 运行各方法（只用共享 unwatermarked 基线，不再重复生成）
# ============================================================
IGSW_K12_CFG = {"algorithm_name": "IGSW", "gamma": 0.5, "delta_ref": 2.0,
    "hash_key": 15485863, "z_threshold": 4.0, "prefix_length": 1,
    "eps": 1e-12, "visualize_mode": "raw_ig",
    "function": "tanh", "delta_min": 1.0, "delta_max": 3.5, "c0": 0.15, "k": 12}

RUNS = [
    ("IGSW", None, "k=12"),
    ("KGW", "config/KGW.json", ""),
    ("SWEET", "config/SWEET.json", ""),
    ("DBW", "config/DBW.json", ""),
]

judger = TimeoutJudger()
os.makedirs("results/tmp_run", exist_ok=True)
all_results = []

for alg_name, config_path, label in RUNS:
    tag = f"{alg_name}_{label}" if label else alg_name
    print(f"\n{'='*60}")
    print(f"Running {tag}")
    print(f"{'='*60}")

    if alg_name == "IGSW" and label == "k=12":
        tmp_path = "results/tmp_run/igsw_k12_mbpp.json"
        with open(tmp_path, "w") as f:
            json.dump(IGSW_K12_CFG, f)
        wm = AutoWatermark.load("IGSW", algorithm_config=tmp_path, transformers_config=tcfg)
    else:
        wm = AutoWatermark.load(alg_name, algorithm_config=config_path, transformers_config=tcfg)

    # PASS@1_wm：只用共享 non_texts，不再重新生成 unwatermarked
    pass_wm_count = 0
    for i in tqdm(range(N), desc=f"{tag} PASS@1"):
        prompt = dataset.get_prompt(i)
        raw_wm = wm.generate_watermarked_text(prompt)
        edited_wm = code_editor.edit(trunc_editor.edit(raw_wm, prompt), prompt)
        ref = dataset.get_reference(i)
        pass_wm_count += judger.analyze(edited_wm, ref)
    pass_wm = pass_wm_count / N
    Q = round(pass_wm / max(pass_non_baseline, 1e-12), 4)
    print(f"  PASS@1_wm={pass_wm:.4f}  Q(shared non={pass_non_baseline:.4f})={Q:.4f}")

    # 生成水印文本
    wm_texts = []
    for i in tqdm(range(N), desc=f"{tag} gen"):
        prompt = dataset.get_prompt(i)
        wm_texts.append(code_editor.edit(
            trunc_editor.edit(wm.generate_watermarked_text(prompt), prompt), prompt))

    # 保存 JSONL
    jsonl_path = f"{LOG_DIR}/{tag.lower().replace(' ','_')}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i in range(N):
            f.write(json.dumps({"idx": i, "prompt": dataset.get_prompt(i),
                                "watermarked_text": wm_texts[i],
                                "unwatermarked_text": non_texts[i]},
                               ensure_ascii=False) + "\n")
    print(f"  Saved: {jsonl_path}")

    # 检测（使用共享 unwatermarked 基线）
    wm_scores, non_scores, skip = [], [], 0
    for i in tqdm(range(N), desc=f"{tag} detect"):
        ok_wm, sw = safe_detect(wm, wm_texts[i])
        ok_non, sn = safe_detect(wm, non_texts[i])
        if ok_wm and ok_non:
            wm_scores.append(sw); non_scores.append(sn)
        else:
            skip += 1

    metrics = compute_metrics(wm_scores, non_scores)
    D = metrics["D"]
    Score = round(2*Q*D/max(Q+D, 1e-12), 4)
    print(f"  AUROC={metrics['AUROC']}  TPR@1%={metrics['TPR@1%']}  F1@1%={metrics['F1@1%']}")
    print(f"  TPR@5%={metrics['TPR@5%']}  F1@5%={metrics['F1@5%']}  Best-F1={metrics['Best-F1']}")
    print(f"  D={D}  Score={Score}  pairs={metrics['pairs']}  skip={skip}")

    all_results.append({
        "algorithm": tag,
        "PASS@1_wm": round(pass_wm, 4),
        "PASS@1_non_shared": round(pass_non_baseline, 4),
        "Q": Q, "D": D, "Score": Score,
        "AUROC": metrics["AUROC"],
        "TPR@1%": metrics["TPR@1%"], "F1@1%": metrics["F1@1%"],
        "TPR@5%": metrics["TPR@5%"], "F1@5%": metrics["F1@5%"],
        "Best-F1": metrics["Best-F1"],
        "pairs": metrics["pairs"], "skipped": skip,
        "jsonl": jsonl_path,
    })

# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 110)
print(f"{'Algorithm':<14} {'P@1_wm':>7} {'P@1_n':>7} {'Q':>6} "
      f"{'AUROC':>7} {'TPR1%':>7} {'F1@1%':>6} {'TPR5%':>7} {'F1@5%':>6} {'BestF1':>7} {'D':>6} {'Score':>6}")
print("-" * 110)
for r in all_results:
    print(f"{r['algorithm']:<14} {r['PASS@1_wm']:>7.4f} {r['PASS@1_non_shared']:>7.4f} {r['Q']:>6.4f} "
          f"{r['AUROC']:>7.4f} {r['TPR@1%']:>7.4f} {r['F1@1%']:>6.4f} "
          f"{r['TPR@5%']:>7.4f} {r['F1@5%']:>6.4f} {r['Best-F1']:>7.4f} {r['D']:>6.4f} {r['Score']:>6.4f}")

summary = {
    "dataset": "MBPP", "model": MODEL_NAME, "max_samples": MAX_SAMPLES,
    "baseline_PASS@1_non": round(pass_non_baseline, 4),
    "params": {"max_new_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE, "top_p": TOP_P},
    "results": all_results,
}
summary_path = f"{LOG_DIR}/summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {summary_path}")
print("Done.")
