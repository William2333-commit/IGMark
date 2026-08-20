"""Robustness comparison plot — WT2 + HumanEval, all attacks."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 13, 'axes.titlesize': 13,
    'legend.fontsize': 9.5, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})

ALGS = ['IGSW', 'DBW', 'KGW', 'SWEET']
COLORS = ['#c44e52', '#dd8452', '#4c72b0', '#55a868']
ATTACKS = ['Word-D', 'Word-S', 'Word-S(C)', 'Translation', 'Doc-P(T5)']
ATTACK_LABELS = ['Word-Del', 'Word-S', 'Word-S(C)', 'Trans', 'Para(T5)']

def load_decays(path):
    with open(path) as f: d = json.load(f)
    results = {}
    for r in d['results']:
        results.setdefault(r['algorithm'], {})[r['attack']] = r['AUROC']
    decays = {}
    for alg in ALGS:
        base = results[alg].get('None', 1)
        decays[alg] = [max(0, (base - results[alg].get(att, base)) / base * 100) for att in ATTACKS]
    return decays

def load_d_values(path):
    with open(path) as f: d = json.load(f)
    results = {}
    for r in d['results']:
        results.setdefault(r['algorithm'], {})[r['attack']] = r['AUROC']
    d_vals = {}
    for alg in ALGS:
        d_vals[alg] = [results[alg].get(att, 0) for att in ATTACKS]
    return d_vals

def load_md_auroc(path):
    """解析 robustness_comparison.md 的 AUROC 表，返回 {alg: [vals...]} 按 ATTACKS 顺序（不含 None）。"""
    with open(path) as f: lines = f.read().splitlines()
    header_idx = next(i for i, ln in enumerate(lines) if 'Attack' in ln and 'AUROC' in ln)
    cols = [c.strip() for c in lines[header_idx].strip('|').split('|')]
    alg_names = [c.replace('AUROC', '').strip() for c in cols[1:]]
    data = {a: [] for a in alg_names}
    for ln in lines[header_idx+2:]:
        if not ln.strip().startswith('|'): break
        parts = [p.strip() for p in ln.strip('|').split('|')]
        if parts[0] == 'None': continue
        for a, v in zip(alg_names, parts[1:]):
            data[a].append(float(v))
    return data

d_vals_wt2 = load_d_values("results/robustness/WT2/robustness.json")
d_vals_he = load_md_auroc("results/robustness/HumanEval/robustness_comparison.md")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

x = np.arange(len(ATTACKS))
width = 0.2

for i, (alg, c) in enumerate(zip(ALGS, COLORS)):
    offset = (i - 1.5) * width
    bars1 = ax1.bar(x + offset, d_vals_wt2[alg], width * 0.92, color=c, alpha=0.82,
                    label=alg, edgecolor='white', linewidth=0.3)
    bars2 = ax2.bar(x + offset, d_vals_he[alg], width * 0.92, color=c, alpha=0.82,
                    edgecolor='white', linewidth=0.3)

ax1.set_xticks(x)
ax1.set_xticklabels(ATTACK_LABELS, rotation=20, ha='right')
ax1.set_ylabel('AUROC', fontsize=13)
ax1.set_title('WT2 (Text)', fontsize=14, fontweight='bold')
ax1.set_ylim(0.80, 1.00)

ax2.set_xticks(x)
ax2.set_xticklabels(ATTACK_LABELS, rotation=20, ha='right')
ax2.set_ylabel('AUROC', fontsize=13)
ax2.set_title('HumanEval (Code)', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 0.90)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
           ncol=4, frameon=True, edgecolor='#ddd', fontsize=9.5)

fig.suptitle('Robustness against Watermark Attacks', fontsize=15, fontweight='bold', y=1.06)
plt.tight_layout()
os.makedirs("results/robustness", exist_ok=True)
plt.savefig("results/robustness/robustness_comparison.pdf", dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig("results/robustness/robustness_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: results/robustness/robustness_comparison.pdf")
plt.close()
