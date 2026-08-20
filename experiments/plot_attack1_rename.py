"""绘制攻击1（变量重命名）鲁棒性对比图：2×2 grouped bar chart。

(a) HumanEval AUROC, (b) MBPP AUROC, (c) HumanEval TPR@5%, (d) MBPP TPR@5%
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

# 字体设置
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# 数据
methods = ['IGSW', 'DBW', 'KGW', 'SWEET']
colors = ['#c44e52', '#dd8452', '#4c72b0', '#55a868']  # red, orange, blue, green

# AUROC
he_auroc_clean  = [0.9052, 0.8523, 0.8004, 0.7872]
he_auroc_rename = [0.8954, 0.8290, 0.7562, 0.7851]
mb_auroc_clean  = [0.9538, 0.8513, 0.7577, 0.7822]
mb_auroc_rename = [0.7727, 0.6776, 0.4034, 0.5344]

# TPR@5%
he_tpr_clean  = [0.7867, 0.5263, 0.3896, 0.4740]
he_tpr_rename = [0.7380, 0.5020, 0.3510, 0.4520]
mb_tpr_clean  = [0.8060, 0.4420, 0.3120, 0.3100]
mb_tpr_rename = [0.4050, 0.2280, 0.0530, 0.1070]

# Δ 值
he_auroc_delta = [c - r for c, r in zip(he_auroc_clean, he_auroc_rename)]
mb_auroc_delta = [c - r for c, r in zip(mb_auroc_clean, mb_auroc_rename)]
he_tpr_delta = [c - r for c, r in zip(he_tpr_clean, he_tpr_rename)]
mb_tpr_delta = [c - r for c, r in zip(mb_tpr_clean, mb_tpr_rename)]

x = np.arange(len(methods))
width = 0.32

fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), constrained_layout=True)

panels = [
    (axes[0, 0], '(a)', 'HumanEval', 'AUROC', he_auroc_clean, he_auroc_rename, he_auroc_delta, (0.35, 1.0)),
    (axes[0, 1], '(b)', 'MBPP', 'AUROC', mb_auroc_clean, mb_auroc_rename, mb_auroc_delta, (0.35, 1.0)),
    (axes[1, 0], '(c)', 'HumanEval', 'TPR@5%FPR', he_tpr_clean, he_tpr_rename, he_tpr_delta, (0.0, 0.85)),
    (axes[1, 1], '(d)', 'MBPP', 'TPR@5%FPR', mb_tpr_clean, mb_tpr_rename, mb_tpr_delta, (0.0, 0.85)),
]

for ax, label, ds, metric, clean, rename, delta, ylim in panels:
    # Clean bars (浅色)
    bars_c = ax.bar(x - width/2, clean, width, color=colors, alpha=0.35, edgecolor=colors, linewidth=1.2, label='Clean')
    # Rename bars (深色实心)
    bars_r = ax.bar(x + width/2, rename, width, color=colors, alpha=0.9, edgecolor='white', linewidth=0.5, label='Rename')

    # IGSW 强调：黑色边框
    bars_r[0].set_edgecolor('#222222')
    bars_r[0].set_linewidth(1.5)

    # Δ 标注（Rename 柱上方）
    for i, (r, d) in enumerate(zip(rename, delta)):
        y_pos = r + (ylim[1] - ylim[0]) * 0.02
        ax.text(x[i] + width/2, y_pos, f'−{d:.3f}', ha='center', va='bottom',
                fontsize=7.5, color='#555555', rotation=0)

    # Clean 柱顶数值
    for i, c in enumerate(clean):
        y_pos = c + (ylim[1] - ylim[0]) * 0.01
        ax.text(x[i] - width/2, y_pos, f'{c:.3f}', ha='center', va='bottom',
                fontsize=7, color='#666666')
    # Rename 柱顶数值（IGSW 加粗）
    for i, r in enumerate(rename):
        y_pos = r - (ylim[1] - ylim[0]) * 0.06
        fw = 'bold' if i == 0 else 'normal'
        ax.text(x[i] + width/2, y_pos, f'{r:.3f}', ha='center', va='top',
                fontsize=7, color='white' if r > 0.5 else '#333333', fontweight=fw)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(ylim)
    ax.set_ylabel(metric)
    ax.set_title(f'{ds}', fontsize=12, fontweight='bold', pad=6)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.axhline(y=0.5, color='#ccc', linewidth=0.5, linestyle='--' if metric == 'AUROC' else '')
    if metric == 'AUROC':
        ax.axhline(y=0.5, color='#bbb', linewidth=0.5, linestyle='--', zorder=0)

    # 子图标签
    ax.text(0.02, 0.95, label, transform=ax.transAxes, fontsize=12, fontweight='bold',
            va='top', ha='left')

# 统一图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='gray', alpha=0.35, edgecolor='gray', linewidth=1.2, label='Clean (no attack)'),
    Patch(facecolor='gray', alpha=0.9, edgecolor='white', linewidth=0.5, label='Rename (attacked)'),
]
fig.legend(handles=legend_elements, loc='upper center', ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 1.02), fontsize=10.5)

# 保存
import os
os.makedirs('results/figures', exist_ok=True)
fig.savefig('results/figures/attack1_rename_robustness.pdf', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig('results/figures/attack1_rename_robustness.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: results/figures/attack1_rename_robustness.pdf / .png")
plt.close()
