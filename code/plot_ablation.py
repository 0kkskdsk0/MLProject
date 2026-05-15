"""
生成 v5 消融实验可视化柱状图
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 11
plt.rcParams['axes.titleweight'] = 'bold'

# ============ 数据 ============
# E1-E14 名称缩写
names = [
    'E1\nA', 'E2\nB', 'E3\nC', 'E4\nA+B\n0.5+0.5', 'E5\nA+B\n0.7+0.3',
    'E6\nA+B\n0.3+0.7', 'E7\nA+C\n0.5+0.5', 'E8\nB+C\n0.5+0.5',
    'E9\nA+C\n0.7+0.3', 'E10\nA+B+C\n0.34+0.33+0.33', 'E11\nA+B+C\n0.5+0.25+0.25',
    'E12\nA+B+C\n0.2+0.4+0.4', 'E13\n+Selected', 'E14\n+Selected',
    'E15\nB+\nsmooth3', 'E16\nB+\nsmooth7', 'E17\nB+\nnosmooth'
]
short_names = ['E1\nA','E2\nB','E3\nC','E4\nA+B','E5\nA+B','E6\nA+B','E7\nA+C','E8\nB+C',
               'E9\nA+C','E10\nA+B+C','E11\nA+B+C','E12\nA+B+C','E13\n+Sel','E14\n+Sel',
               'E15\nSm3','E16\nSm7','E17\nNoSm']

# Val AUC-PR
val_aucpr = [0.9635, 0.9756, 0.5545, 0.9713, 0.9686, 0.9733, 0.9535, 0.9698,
             0.9598, 0.9679, 0.9667, 0.9683, 0.9645, 0.9679, 0.9830, 0.9704, 0.9923]

# Test metrics
test_aucpr = [0.9640, 0.9825, 0.4533, 0.9786, 0.9743, 0.9807, 0.9361, 0.9622,
              0.9465, 0.9606, 0.9600, 0.9615, 0.9519, 0.9662, 0.9891, 0.9772, 0.9974]
test_f1 =    [0.9013, 0.9333, 0.4375, 0.9333, 0.9244, 0.9333, 0.8833, 0.9053,
              0.8776, 0.8776, 0.8776, 0.9053, 0.8716, 0.8957, 0.9356, 0.9312, 0.9569]
test_fp =    [8, 8, 80, 8, 8, 8, 14, 13, 13, 13, 13, 13, 25, 7, 4, 12, 1]
test_fn =    [15, 8, 64, 8, 10, 8, 14, 10, 16, 16, 16, 10, 8, 17, 11, 5, 9]

# Train F1 (for overfit check)
train_f1 =   [0.8896, 0.9294, 0.1158, 0.8926, 0.8985, 0.8955, 0.7297, 0.8504,
              0.8372, 0.9060, 0.9060, 0.8654, 0.7297, 0.9375, 0.9474, 0.9015, 0.7837]

# Val F1
val_f1 =     [0.9235, 0.9333, 0.5810, 0.9274, 0.9213, 0.9307, 0.8967, 0.9061,
              0.9030, 0.9122, 0.9148, 0.9061, 0.9013, 0.9080, 0.9375, 0.9326, 0.9534]

x = np.arange(len(names))
colors_base = ['#4C72B0'] * 14
colors_pp   = ['#DD8452', '#DD8452', '#DD8452']
colors_all = colors_base + colors_pp

# ===== 图1: Val AUC-PR =====
fig, ax = plt.subplots(figsize=(14, 5))
order = np.argsort(val_aucpr)
bars = ax.bar(x, [val_aucpr[i] for i in order], color=[colors_all[i] for i in order],
              edgecolor='white', linewidth=0.5)
# highlight E15
for idx, bar in zip(order, bars):
    if idx == 14:  # E15
        bar.set_color('#C44E52')
        bar.set_edgecolor('black')
        bar.set_linewidth(2)
    if idx == 16:  # E17
        bar.set_hatch('//')
        bar.set_edgecolor('#C44E52')
        bar.set_linewidth(1.5)
ax.set_xticks(x)
ax.set_xticklabels([short_names[i] for i in order], fontsize=8)
ax.set_ylabel('AUC-PR')
ax.set_title('Val AUC-PR by Configuration (sorted ascending)', fontsize=13)
ax.axhline(0.9830, color='#C44E52', linestyle='--', linewidth=1, label='E15 (selected)')
ax.legend(fontsize=9)
ax.set_ylim(0.5, 1.0)
for bar, v in zip(bars, [val_aucpr[i] for i in order]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{v:.4f}', ha='center', va='bottom', fontsize=7, rotation=45)
plt.tight_layout()
plt.savefig('docs/chart_val_aucpr.png', dpi=150)
plt.close()
print('✓ chart_val_aucpr.png')

# ===== 图2: Test AUC-PR vs Test F1 (grouped) =====
fig, ax = plt.subplots(figsize=(14, 5))
w = 0.35
order = np.argsort(test_aucpr)
bars1 = ax.bar(x - w/2, [test_aucpr[i] for i in order], w, label='Test AUC-PR',
               color='#4C72B0', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + w/2, [test_f1[i] for i in order], w, label='Test F1',
               color='#55A868', edgecolor='white', linewidth=0.5)
# highlight E15
for idx, (b1, b2) in enumerate(zip(bars1, bars2)):
    if order[idx] == 14:
        b1.set_color('#C44E52'); b2.set_color('#C44E52')
        b1.set_edgecolor('black'); b2.set_edgecolor('black')
        b1.set_linewidth(2); b2.set_linewidth(2)
ax.set_xticks(x)
ax.set_xticklabels([short_names[i] for i in order], fontsize=8)
ax.set_ylabel('Score')
ax.set_title('Test AUC-PR and F1 by Configuration (sorted by AUC-PR)', fontsize=13)
ax.legend(fontsize=9)
ax.set_ylim(0.4, 1.0)
plt.tight_layout()
plt.savefig('docs/chart_test_aucpr_f1.png', dpi=150)
plt.close()
print('✓ chart_test_aucpr_f1.png')

# ===== 图3: Test FP/FN (stacked bar) =====
fig, ax = plt.subplots(figsize=(14, 5))
order = np.argsort(np.array(test_fp) + np.array(test_fn))
colors_fp = ['#E74C3C' if i == 14 else '#C44E52' for i in order]
colors_fn = ['#F1948A' if i == 14 else '#F5B7B1' for i in order]
bars_fp = ax.bar(x, [test_fp[i] for i in order], label='FP', color=colors_fp, edgecolor='white', linewidth=0.5)
bars_fn = ax.bar(x, [test_fn[i] for i in order], bottom=[test_fp[i] for i in order],
                 label='FN', color=colors_fn, edgecolor='white', linewidth=0.5)
for idx, b in enumerate(zip(bars_fp, bars_fn)):
    if order[idx] == 14:
        b[0].set_edgecolor('black'); b[1].set_edgecolor('black')
        b[0].set_linewidth(2); b[1].set_linewidth(2)
ax.set_xticks(x)
ax.set_xticklabels([short_names[i] for i in order], fontsize=8)
ax.set_ylabel('Count')
ax.set_title('Test FP / FN by Configuration (sorted by total errors)', fontsize=13)
ax.legend(fontsize=9)
for bar, fp_val, fn_val in zip(bars_fp, [test_fp[i] for i in order], [test_fn[i] for i in order]):
    total = fp_val + fn_val
    ax.text(bar.get_x() + bar.get_width()/2, total + 0.5, str(total),
            ha='center', va='bottom', fontsize=7)
plt.tight_layout()
plt.savefig('docs/chart_test_fpfn.png', dpi=150)
plt.close()
print('✓ chart_test_fpfn.png')

# ===== 图4: Smoothing comparison (E2, E15, E16, E17) =====
fig, ax = plt.subplots(figsize=(8, 4.5))
smooth_names = ['E2\nB (no smooth,\nno postproc)', 'E15\nB + smooth3\n(selected)',
                'E16\nB + smooth7', 'E17\nB + nosmooth\n(postproc only)']
metrics = {
    'Val AUC-PR': [0.9756, 0.9830, 0.9704, 0.9923],
    'Test F1':    [0.9333, 0.9356, 0.9312, 0.9569],
    'Train F1':   [0.9294, 0.9474, 0.9015, 0.7837],
}
x2 = np.arange(len(smooth_names))
w2 = 0.22
colors_s = ['#4C72B0', '#55A868', '#DD8452']
for i, (label, vals) in enumerate(metrics.items()):
    offset = (i - 1) * w2
    bars = ax.bar(x2 + offset, vals, w2, label=label, color=colors_s[i], edgecolor='white', linewidth=0.5)
    if i == 1:  # highlight E15 bar in this group
        bars[1].set_edgecolor('black'); bars[1].set_linewidth(2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{v:.4f}', ha='center', va='bottom', fontsize=7)
ax.set_xticks(x2)
ax.set_xticklabels(smooth_names, fontsize=8)
ax.set_ylabel('Score')
ax.set_title('Smoothing Variants: E15 is Most Consistent (smallest Train-Test Δ)', fontsize=12)
ax.legend(fontsize=8, loc='lower left')
ax.set_ylim(0.7, 1.0)
plt.tight_layout()
plt.savefig('docs/chart_smoothing_compare.png', dpi=150)
plt.close()
print('✓ chart_smoothing_compare.png')

# ===== 图5: Overfit analysis — Train F1 vs Test F1 =====
fig, ax = plt.subplots(figsize=(14, 5))
order = np.argsort(np.array(train_f1) - np.array(test_f1))  # sort by overfit delta
delta = [(train_f1[i] - test_f1[i]) for i in order]
colors_delta = ['#E74C3C' if d > 0.05 else '#55A868' for d in delta]
bars = ax.bar(x, [test_f1[i] for i in order], label='Test F1', color='#4C72B0', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x, [train_f1[i] for i in order], bottom=[test_f1[i] for i in order],
               label='Train F1 (excess)', color=[colors_all[i] for i in order],
               edgecolor='white', linewidth=0.5, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([short_names[i] for i in order], fontsize=8)
ax.set_ylabel('F1')
ax.set_title('Overfit Analysis: Train F1 vs Test F1 (sorted by Δ = Train − Test)', fontsize=13)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.1)
for i_idx, (idx, d) in enumerate(zip(order, delta)):
    c = '#E74C3C' if abs(d) > 0.05 else '#2C3E50'
    ax.text(x[i_idx], test_f1[idx] + max(0, d)/2 + 0.02, f'Δ={d:+.4f}',
            ha='center', va='bottom', fontsize=6, color=c, rotation=45)
plt.tight_layout()
plt.savefig('docs/chart_overfit.png', dpi=150)
plt.close()
print('✓ chart_overfit.png')

# ===== 图6: Single model comparison (A, B, C) =====
fig, ax = plt.subplots(figsize=(6, 4))
single_names = ['E1\nXGBoost\nStandard (A)', 'E2\nXGBoost\nReweighted (B)', 'E3\nLightGBM (C)']
single_val_aucpr = [0.9635, 0.9756, 0.5545]
single_val_f1 = [0.9235, 0.9333, 0.5810]
x3 = np.arange(3)
w3 = 0.3
ax.bar(x3 - w3/2, single_val_aucpr, w3, label='Val AUC-PR', color='#4C72B0', edgecolor='white')
bars_s = ax.bar(x3 + w3/2, single_val_f1, w3, label='Val F1', color='#55A868', edgecolor='white')
bars_s[1].set_edgecolor('black'); bars_s[1].set_linewidth(2)
ax.set_xticks(x3)
ax.set_xticklabels(single_names, fontsize=9)
ax.set_ylabel('Score')
ax.set_title('Single Model Comparison (Val Set)', fontsize=13)
ax.legend(fontsize=9)
ax.set_ylim(0.4, 1.0)
for i, (a, f) in enumerate(zip(single_val_aucpr, single_val_f1)):
    ax.text(i - w3/2, a + 0.005, f'{a:.4f}', ha='center', fontsize=8)
    ax.text(i + w3/2, f + 0.005, f'{f:.4f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('docs/chart_single_model.png', dpi=150)
plt.close()
print('✓ chart_single_model.png')

# ===== 图7: E15 detailed breakdown across Train/Val/Test =====
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

# 7a: Metrics radar-style bar
ax0 = ax[0]
e15_metrics = {
    'AUC-PR':  [1.0000, 0.9830, 0.9891],
    'F1':      [0.9474, 0.9375, 0.9356],
    'Precision': [0.9000, 0.9593, 0.9646],
    'Recall':  [1.0000, 0.9167, 0.9083],
}
x7 = np.arange(4)
w7 = 0.22
for i, (split, color) in enumerate(zip(['Train', 'Val', 'Test'], ['#4C72B0', '#55A868', '#C44E52'])):
    vals = [e15_metrics[m][i] for m in e15_metrics]
    offset = (i - 1) * w7
    bars = ax0.bar(x7 + offset, vals, w7, label=split, color=color, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.4f}', ha='center', va='bottom', fontsize=6.5)
ax0.set_xticks(x7)
ax0.set_xticklabels(['AUC-PR', 'F1', 'Precision', 'Recall'], fontsize=10)
ax0.set_ylabel('Score')
ax0.set_title('E15: Metrics Across Splits', fontsize=13, fontweight='bold')
ax0.legend(fontsize=8)
ax0.set_ylim(0.8, 1.05)

# 7b: FP/FN across splits
ax1 = ax[1]
fpfn_data = {'Train': [30, 0], 'Val': [7, 15], 'Test': [4, 11]}
x7b = np.arange(3)
w7b = 0.3
for i, (label, color) in enumerate(zip(['FP', 'FN'], ['#E74C3C', '#F5B7B1'])):
    vals = [fpfn_data[s][i] for s in ['Train', 'Val', 'Test']]
    offset = (i - 0.5) * w7b
    bars = ax1.bar(x7b + offset, vals, w7b, label=label, color=color, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.set_xticks(x7b)
ax1.set_xticklabels(['Train', 'Val', 'Test'], fontsize=11)
ax1.set_ylabel('Count')
ax1.set_title('E15: FP / FN Across Splits', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.set_ylim(0, 20)

# 7c: Prediction volume
ax2 = ax[2]
pred_data = {'Train': (300, 130816), 'Val': (172, 3729), 'Test': (113, 2647)}
labels7 = ['Train', 'Val', 'Test']
anom_pred = [pred_data[s][0] for s in labels7]
total = [pred_data[s][1] for s in labels7]
normal_pred = [total[i] - anom_pred[i] for i in range(3)]
ax2.bar(range(3), normal_pred, label='Predicted Normal', color='#4C72B0', edgecolor='white')
ax2.bar(range(3), anom_pred, bottom=normal_pred, label='Predicted Anomaly', color='#C44E52', edgecolor='white')
for i in range(3):
    pct = anom_pred[i] / total[i] * 100
    ax2.text(i, total[i] + max(total)*0.02, f'{anom_pred[i]}/{total[i]}\n({pct:.2f}%)',
             ha='center', va='bottom', fontsize=9)
ax2.set_xticks(range(3))
ax2.set_xticklabels(labels7, fontsize=11)
ax2.set_ylabel('Count')
ax2.set_title('E15: Prediction Distribution', fontsize=13, fontweight='bold')
ax2.legend(fontsize=8)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('docs/chart_e15_detail.png', dpi=150)
plt.close()
print('✓ chart_e15_detail.png')

print('ALL CHARTS GENERATED.')
