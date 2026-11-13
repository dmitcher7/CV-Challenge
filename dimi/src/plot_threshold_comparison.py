"""
Genereert een complete visuele vergelijkingsfiguur van alle 18 threshold configuraties.
Auteur: Dmitrij Tchervonisjenko (Team Git Push and Pray)
Thomas More Geel - Deep Learning & Computer Vision
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
CSV_PATH = ROOT_DIR / "config" / "iou_nms_benchmark.csv"
OUT_PATH = ROOT_DIR / "runs" / "threshold_comparison_charts.png"

df = pd.read_csv(CSV_PATH)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

labels = [
    f"{r['Config Name'].split('(')[0].strip()[:14]} ({r['Conf Threshold']}/{r['IoU Threshold']})"
    for _, r in df.iterrows()
]
x = range(len(df))
w = 0.38
win_idx = df['mAP@50'].idxmax()

# 1. Barplot mAP@50 & mAP@50-95
bar1 = axes[0, 0].bar([i - w/2 for i in x], df['mAP@50'], width=w, label='mAP@50', color='#1f77b4', alpha=0.85)
bar2 = axes[0, 0].bar([i + w/2 for i in x], df['mAP@50-95'], width=w, label='mAP@50-95', color='#ff7f0e', alpha=0.85)
bar1[win_idx].set_color('#00a86b')
bar2[win_idx].set_color('#2e8b57')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
axes[0, 0].set_title('1. mAP@50 & mAP@50-95 per Threshold Configuratie (Groen = Winnaar)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Score')
axes[0, 0].legend()
axes[0, 0].grid(True, linestyle='--', alpha=0.6)

# 2. Precision vs Recall Trade-Off
sc = axes[0, 1].scatter(
    df['Recall (R)'], df['Precision (P)'],
    s=df['mAP@50'] * 500, c=df['F1-Score'],
    cmap='viridis', edgecolors='black', linewidth=1.5, alpha=0.9
)
cb = plt.colorbar(sc, ax=axes[0, 1])
cb.set_label('F1-Score')
for i, r in df.iterrows():
    axes[0, 1].annotate(
        labels[i], (r['Recall (R)'] + 0.001, r['Precision (P)'] + 0.001),
        fontsize=8, fontweight='bold' if i == win_idx else 'normal'
    )
axes[0, 1].set_title('2. Precision vs. Recall Trade-Off (Cirkelgrootte = mAP@50)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Recall (R)')
axes[0, 1].set_ylabel('Precision (P)')
axes[0, 1].grid(True, linestyle='--', alpha=0.6)

# 3. Bal mAP@50 vs Confidence Threshold
df_sorted_conf = df.sort_values(by='Conf Threshold')
axes[1, 0].plot(df_sorted_conf['Conf Threshold'], df_sorted_conf['Ball mAP@50'], marker='o', color='#d62728', linewidth=2.5, markersize=8, label='Bal mAP@50')
axes[1, 0].plot(df_sorted_conf['Conf Threshold'], df_sorted_conf['mAP@50'], marker='s', color='#1f77b4', linewidth=2.0, linestyle='--', markersize=6, label='Overall mAP@50')
axes[1, 0].set_xscale('log')
axes[1, 0].set_title('3. Impact van Confidence Drempel (Log Schaal) op Bal mAP@50 & Overall mAP', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Confidence Threshold (log schaal)')
axes[1, 0].set_ylabel('mAP Score')
axes[1, 0].legend()
axes[1, 0].grid(True, linestyle='--', alpha=0.6)

# 4. IoU NMS Threshold Sensitivity (0.50 vs 0.60 vs 0.70)
sub = df[df['Conf Threshold'].isin([0.005, 0.010, 0.025])]
for c_val, group in sub.groupby('Conf Threshold'):
    sorted_group = group.sort_values(by='IoU Threshold')
    axes[1, 1].plot(sorted_group['IoU Threshold'], sorted_group['mAP@50'], marker='^', linewidth=2, markersize=8, label=f'conf={c_val}')
axes[1, 1].set_title('4. IoU NMS Gevoeligheid (0.50 vs 0.60 vs 0.70)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('NMS IoU Threshold')
axes[1, 1].set_ylabel('mAP@50 Score')
axes[1, 1].legend()
axes[1, 1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=300)
print(f"[SUCCESS] Figuur succesvol opgeslagen op: {OUT_PATH}")
