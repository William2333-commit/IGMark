"""IGWD weight function comparison plot — one dataset per figure."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.labelsize': 13, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})

FUNCTIONS = ['EWD', 'Linear', 'Sigmoid', 'Exponential', 'Tanh']
ALGS = ['DBW']

lin_he = {"IGSW":0.9404,"SWEET":0.5953,"DBW":0.9329}
lin_mb = {"IGSW":0.9022,"SWEET":0.384,"DBW":0.8324}
ewd_he = {"IGSW":0.792,"SWEET":0.6518,"DBW":0.7521}
ewd_mb = {"IGSW":0.6711,"SWEET":0.347,"DBW":0.5854}

DATASETS = {
    "HumanEval": {"color": 'Greens', "cmap_colors": ['#27ae60', '#2ecc71', '#a9dfbf', '#82e0aa', '#58d68d'],
                  "lin": lin_he, "ewd": ewd_he},
    "MBPP": {"color": 'Blues', "cmap_colors": ['#2980b9', '#3498db', '#aed6f1', '#85c1e9', '#5dade2'],
             "lin": lin_mb, "ewd": ewd_mb},
}

DATA_ALL = {}

for ds, cfg in DATASETS.items():
    with open(f"results/igwd_functions/{ds}/results.json") as f:
        d = json.load(f)
    results = {}
    for r in d['results']:
        results.setdefault(r['generator'], {})[r['weight_func']] = r['F1@1%']
    DATA_ALL[ds] = {'results': results, 'ewd': cfg['ewd'], 'lin': cfg['lin']}

# ---- Single figure: HumanEval-DBW + MBPP-DBW side by side ----
fig, ax = plt.subplots(figsize=(7, 3.8))
x = np.array([0, 0.15])
spacing = 0.02
bar_w = 0.02

fn_colors = ['#d4e6f1', '#85c1e9', '#3498db', '#2980b9', '#1a5276']  # blue family

for i, fn in enumerate(FUNCTIONS):
    values = []
    for ds in ['HumanEval', 'MBPP']:
        cfg = DATA_ALL[ds]
        if fn == 'EWD': v = cfg['ewd'].get('DBW', 0)
        elif fn == 'Linear': v = cfg['lin'].get('DBW', 0)
        else: v = cfg['results'].get('DBW', {}).get(fn, 0)
        values.append(v)

    offset = (i - 2) * spacing
    ax.bar(x + offset, values, bar_w, color=fn_colors[i],
           label=fn, edgecolor='white', linewidth=0.3)

ax.set_xticks(x)
ax.set_xticklabels(['HumanEval', 'MBPP'], fontsize=13)
ax.set_xlim(-0.08, 0.23)
ax.set_ylabel('F1 @ 1% FPR', fontsize=12)
ax.set_ylim(0, 1.08)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), frameon=True,
          edgecolor='#ddd', fontsize=9, ncol=5)
plt.tight_layout()
plt.savefig("results/igwd_functions/sweet_comparison.pdf", dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig("results/igwd_functions/sweet_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: sweet_comparison.pdf")
