"""Robustness evaluation: attack → detect → compare before/after.

Usage:
  CUDA_VISIBLE_DEVICES=X python experiments/run_robustness.py HumanEval
  CUDA_VISIBLE_DEVICES=X python experiments/run_robustness.py WT2

Attacks: Word-D, Word-S, Word-S(Context), Translation, Doc-P(Dipper)
"""
import os, sys, json, math, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch, numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

from transformers import (AutoTokenizer, AutoModelForCausalLM,
    T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForMaskedLM)
from translate import Translator

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.tools.text_editor import (
    WordDeletion, SynonymSubstitution, ContextAwareSynonymSubstitution,
    DipperParaphraser, BackTranslationTextEditor)
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
from evaluation.pipelines.detection import (WatermarkedTextDetectionPipeline,
    UnWatermarkedTextDetectionPipeline, DetectionPipelineReturnType)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DS = sys.argv[1] if len(sys.argv) > 1 else "WT2"

TARGET_DS_FOLDER = {
    "HumanEval": "results/humaneval/compare",
    "WT2": "results/wt2/sample_t0.7",
}
FOLDER = TARGET_DS_FOLDER[DS]

MODEL_MAP = {
    "HumanEval": "bigcode/starcoder",
    "WT2": "facebook/opt-1.3b",
}
MODEL_NAME = MODEL_MAP[DS]

GEN_KWARGS = {
    "HumanEval": dict(max_new_tokens=512, do_sample=True, temperature=0.2, top_p=0.95),
    "WT2": dict(max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.95),
}[DS]

# Watermark configs
HASH = 15485863
IGSW_CFG = {"algorithm_name": "IGSW", "gamma": 0.5, "delta_ref": 2.0, "hash_key": HASH,
    "z_threshold": 4.0, "prefix_length": 1, "eps": 1e-12, "visualize_mode": "raw_ig",
    "function": "tanh", "delta_min": 1.0, "delta_max": 3.0, "c0": 0.15, "k": 12}
SWEET_CFG = {"algorithm_name": "SWEET", "gamma": 0.5, "delta": 2.0, "hash_key": HASH,
    "z_threshold": 4.0, "prefix_length": 1, "entropy_threshold": 0.65}

RESULTS = {}
PID = os.getpid()
LOG_DIR = f"results/robustness/{DS}"
os.makedirs(LOG_DIR, exist_ok=True)

print(f"[PID={PID}] Robustness: {DS} | {MODEL_NAME}")

# ========== Load model + watermarks ==========
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True).to("cuda:0")
model.eval()
tcfg = TransformersConfig(model=model, tokenizer=tokenizer, vocab_size=len(tokenizer), device="cuda", **GEN_KWARGS)

def load_wm(alg_key):
    if alg_key == "IGSW":
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(IGSW_CFG, tmp); tmp.close()
        wm = AutoWatermark.load("IGSW", algorithm_config=tmp.name, transformers_config=tcfg)
        os.unlink(tmp.name)
    elif alg_key == "SWEET":
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(SWEET_CFG, tmp); tmp.close()
        wm = AutoWatermark.load("SWEET", algorithm_config=tmp.name, transformers_config=tcfg)
        os.unlink(tmp.name)
    else:
        wm = AutoWatermark.load(alg_key, algorithm_config=f"config/{alg_key}.json", transformers_config=tcfg)
    return wm

wms = {alg: load_wm(alg) for alg in ["IGSW", "KGW", "SWEET", "DBW"]}

# ========== Load texts ==========
print("Loading texts...")
# Baseline
if DS == "WT2":
    with open(f"{FOLDER}/baseline.jsonl") as f:
        bdata = [json.loads(l) for l in f]
    non_texts = [d["unwatermarked_text"] for d in bdata]
elif DS == "HumanEval":
    igsw_file = next(f for f in os.listdir(FOLDER) if f.startswith("igsw") and f.endswith(".jsonl"))
    with open(f"{FOLDER}/{igsw_file}") as f:
        bdata = [json.loads(l) for l in f]
    non_texts = [d["unwatermarked_text"] for d in bdata]

N = min(len(non_texts), 300)  # use up to 300 for speed
non_texts = non_texts[:N]
print(f"  Non: {len(non_texts)}")

# Watermarked texts
wm_texts = {}
wm_prompts = {}
for alg_key, (alg_name, suffix) in {
    "IGSW": ("IGSW", "igsw"),
    "KGW": ("KGW", "kgw"),
    "SWEET": ("SWEET", "sweet"),
    "DBW": ("DBW", "dbw"),
}.items():
    if DS == "HumanEval":
        file = next((f for f in os.listdir(FOLDER) if f.startswith(suffix) and f.endswith(".jsonl")), None)
    else:
        file = next((f for f in os.listdir(FOLDER) if f.startswith(suffix) and f.endswith(".jsonl")), None)
    if file:
        with open(f"{FOLDER}/{file}") as f:
            data = [json.loads(l) for l in f]
        wm_texts[alg_key] = [d.get("watermarked_text", d.get("watermarked", "")) for d in data][:N]
        wm_prompts[alg_key] = [d.get("prompt", "") for d in data][:N]
        print(f"  {alg_key}: {len(wm_texts[alg_key])}")
    else:
        print(f"  {alg_key}: NOT FOUND")

# ========== Attacks ==========
print("Setting up attacks...")

attacks = {
    "None": None,  # no attack baseline
    "Word-D": WordDeletion(ratio=0.3),
}

# WordNet synonym (Word-S)
try:
    attacks["Word-S"] = SynonymSubstitution(ratio=0.5)
except:
    print("  Word-S: failed to init")

# BERT context-aware synonym (Word-S(Context))
try:
    bert_tok = BertTokenizer.from_pretrained("bert-large-uncased")
    bert_model = BertForMaskedLM.from_pretrained("bert-large-uncased").to("cuda:0")
    bert_model.eval()
    attacks["Word-S(C)"] = ContextAwareSynonymSubstitution(ratio=0.5, tokenizer=bert_tok, model=bert_model, device="cuda")
except Exception as e:
    print(f"  Word-S(C): {e}")

# Translation (en->de->en) using local Helsinki-NLP OPUS-MT models
try:
    from transformers import MarianMTModel, MarianTokenizer
    en_de_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    en_de_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de").to("cuda:0")
    de_en_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    de_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en").to("cuda:0")

    def translate_en_de(text):
        if not text.strip(): return text
        enc = en_de_tok(text[:500], return_tensors="pt", truncation=True, max_length=512).to("cuda:0")
        out = en_de_model.generate(**enc, max_new_tokens=512)
        return en_de_tok.decode(out[0], skip_special_tokens=True)

    def translate_de_en(text):
        if not text.strip(): return text
        enc = de_en_tok(text[:500], return_tensors="pt", truncation=True, max_length=512).to("cuda:0")
        out = de_en_model.generate(**enc, max_new_tokens=512)
        return de_en_tok.decode(out[0], skip_special_tokens=True)

    attacks["Translation"] = BackTranslationTextEditor(
        translate_to_intermediary=translate_en_de,
        translate_to_source=translate_de_en)
except Exception as e:
    print(f"  Translation: {e}")

# T5-based paraphraser (replaces broken DIPPER)
class T5Paraphraser:
    def __init__(self, tok, model, device="cuda"):
        self.tok = tok; self.model = model; self.device = device
    def edit(self, text, reference=None):
        if not text.strip(): return text
        inp = f"paraphrase: {text[:500]}"
        enc = self.tok([inp], return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=300, do_sample=True, temperature=1.0, top_p=0.9)
        return self.tok.decode(out[0], skip_special_tokens=True)

try:
    from transformers import AutoTokenizer as AT, AutoModelForSeq2SeqLM
    p_tok = AT.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
    p_model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base", torch_dtype=torch.float16).to("cuda:0")
    p_model.eval()
    attacks["Doc-P(T5)"] = T5Paraphraser(tok=p_tok, model=p_model)
    print("  Doc-P(T5): loaded")
except Exception as e:
    print(f"  Doc-P(T5): {e}")

print(f"  Attacks ready: {list(attacks.keys())}")

# ========== Detection Pipeline ==========
def compute_metrics(wm_scores, non_scores):
    if len(wm_scores) < 2 or len(non_scores) < 2:
        return {"AUROC": 0, "TPR@1%": 0, "F1@1%": 0, "TPR@5%": 0, "F1@5%": 0, "Best-F1": 0, "D": 0, "pairs": 0}
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
    D = round(float((auroc+tpr5)/2), 4)
    return {"AUROC": round(auroc, 4), "TPR@1%": round(tpr1, 4), "F1@1%": round(f1_1, 4),
            "TPR@5%": round(tpr5, 4), "F1@5%": round(f1_5, 4), "Best-F1": round(best_f1, 4),
            "D": D, "pairs": len(wm_scores)}

def parse_score(r):
    if isinstance(r, dict):
        for k in ["score", "z_score", "z"]:
            if k in r: return float(r[k])
    if isinstance(r, (int, float)): return float(r)
    return None

def safe_detect(wm, text):
    if not text or len(text.strip()) == 0: return None
    try:
        s = parse_score(wm.detect_watermark(text, return_dict=True))
        return s if s is not None and not math.isnan(s) and not math.isinf(s) else None
    except: return None

# ========== Run ==========
all_results = []

for alg_key in ["IGSW", "KGW", "SWEET", "DBW"]:
    if alg_key not in wm_texts: continue
    wm = wms[alg_key]
    texts = wm_texts[alg_key]
    print(f"\n{'='*50}\n{alg_key}\n{'='*50}")

    for att_name, attack in attacks.items():
        print(f"  {att_name}...")
        if att_name == "None":
            attacked = texts
        else:
            attacked = []
            for i, t in enumerate(tqdm(texts, desc=f"{alg_key}/{att_name}")):
                try:
                    # T5/DIPPER needs prompt as reference
                    if att_name in ("Doc-P(T5)", "Doc-P(Dipper)"):
                        prompt_ref = wm_prompts[alg_key][i] if alg_key in wm_prompts else ""
                        try:
                            attacked.append(attack.edit(t, prompt_ref))
                        except:
                            attacked.append(attack.edit(t))
                    else:
                        attacked.append(attack.edit(t))
                except:
                    attacked.append(t)  # fallback: keep original

        # Detect
        wm_scores = [s for t in tqdm(attacked, desc=f"detect wm") if (s := safe_detect(wm, t)) is not None]
        non_scores = [s for t in tqdm(non_texts[:len(attacked)], desc=f"detect non") if (s := safe_detect(wm, t)) is not None]
        n = min(len(wm_scores), len(non_scores))
        m = compute_metrics(wm_scores[:n], non_scores[:n])
        m["algorithm"] = alg_key; m["attack"] = att_name
        print(f"    D={m['D']:.4f} AUROC={m['AUROC']:.4f} TPR1={m['TPR@1%']:.4f} TPR5={m['TPR@5%']:.4f} p={m['pairs']}")
        all_results.append(m)

# ========== Save ==========
summary = {"dataset": DS, "model": MODEL_NAME, "samples": N, "results": all_results}
with open(f"{LOG_DIR}/robustness.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# Summary table
print(f"\n{'='*100}")
print(f"{'Alg':<6} {'Attack':<16} {'AUROC':>7} {'TPR1':>7} {'F1@1':>7} {'TPR5':>7} {'F1@5':>7} {'BF1':>7} {'D':>7} {'D_decay':>8}")
print("-"*100)
results_by_alg = {}
for r in all_results:
    results_by_alg.setdefault(r["algorithm"], {})
    results_by_alg[r["algorithm"]][r["attack"]] = r

for alg in ["IGSW", "KGW", "SWEET", "DBW"]:
    if alg not in results_by_alg: continue
    base_D = results_by_alg[alg].get("None", {}).get("D", 1.0)
    for att in ["None", "Word-D", "Word-S", "Word-S(C)", "Translation", "Doc-P(T5)"]:
        r = results_by_alg[alg].get(att)
        if r is None: continue
        decay = f"{(base_D - r['D'])/base_D*100:.0f}%" if base_D > 0 else "-"
        print(f"{alg:<6} {att:<16} {r['AUROC']:>7.4f} {r['TPR@1%']:>7.4f} {r['F1@1%']:>7.4f} {r['TPR@5%']:>7.4f} {r['F1@5%']:>7.4f} {r['Best-F1']:>7.4f} {r['D']:>7.4f} {decay:>8}")

print(f"\nSaved: {LOG_DIR}/robustness.json\nDone!")
