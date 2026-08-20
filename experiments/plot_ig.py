"""IG distribution — clean bars, few bins, key x-axis points."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 13, 'axes.titlesize': 14,
    'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})

DATA_DIR = "results/analysis"
K, C0, DMIN, DMAX = 12, 0.15, 1.0, 3.0

fig = plt.figure(figsize=(7, 3.5))
ax = fig.add_subplot(111)

# ====== IG histogram (few bins) ======
BINS = np.linspace(0, 0.5, 51)
colors = ['#e74c3c', '#e67e22', '#27ae60', '#2980b9']
labels = ['HumanEval', 'MBPP', 'WMT', 'C4']
bar_width = (BINS[1]-BINS[0]) / 5

for i, (ds, c, lbl) in enumerate(zip(['HumanEval','MBPP','WMT','C4'], colors, labels)):
    ig = np.clip(np.load(f"{DATA_DIR}/{ds}_ig.npz")['ig'], 0, 0.5)
    counts, _ = np.histogram(ig, bins=BINS)
    pct = counts / counts.sum() * 100
    offset = (i - 1.5) * bar_width
    bars = ax.bar(BINS[:-1] + offset, pct, width=bar_width * 0.95, color=c, alpha=0.78,
                  label=lbl, edgecolor='white', linewidth=0.3)

# Key IG reference line
ax.axvline(C0, color='#333', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(C0+0.008, 2.85, f'$c_0$={C0}', fontsize=11, color='#333', fontweight='bold')

ax.set_xlabel('Information Gain (IG)', fontsize=14)
ax.set_ylabel('Percentage', fontsize=14)
ax.set_xlim(0, 0.5)
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_ylim(0, 3.2)
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.legend(loc='upper right', frameon=True, edgecolor='#ccc', fontsize=10)

os.makedirs("results/analysis", exist_ok=True)
plt.savefig("results/analysis/ig_distribution.pdf", dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig("results/analysis/ig_distribution.png", dpi=300, bbox_inches='tight', facecolor='white')
print("Saved")
plt.close()
