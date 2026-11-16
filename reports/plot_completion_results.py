"""Plot completed independent-test sweep and SNR evidence."""
from pathlib import Path
import csv
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
cohort = root/'final-report-work/methods/results/course_completion_20261109-133839'
output = cohort/'plots'
output.mkdir(exist_ok=True)
rows = list(csv.DictReader((cohort/'independent_evaluation.csv').open()))
pattern = re.compile(r'fraction_sl(\d+)_total85000_seed(\d+)\.npz$')
data = []
for row in rows:
    match = pattern.search(row['source'])
    if match and row['stage'] in ['stage2_sweep_evaluation', 'stage3_snr'] and row['status'] == 'ok':
        row = dict(row, supervised=int(match[1]), seed=int(match[2]))
        for key in ['pmd', 'pfa', 'ser', 'rmse_detected', 'sensing_snr_db']:
            row[key] = float(row[key])
        row['pmd_ci95'] = json.loads(row['pmd_ci95'])
        data.append(row)
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
sweep = sorted([r for r in data if r['stage'] == 'stage2_sweep_evaluation'], key=lambda r:r['supervised'])
assert len(sweep) == 5
fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
for axis, key, label, scale in zip(axes,
        ['pmd', 'rmse_detected', 'ser'],
        ['Misdetection probability [%]', 'Detected-target position RMSE [m]', 'Symbol error rate [%]'],
        [100, 1, 100]):
    ssl = [r for r in sweep if r['seed'] == 42]
    budgets = [0,100,500,1000,5000]
    x = [budgets.index(r['supervised']) for r in ssl]
    y = np.array([r[key] for r in ssl])*scale
    if key == 'pmd':
        limits = np.array([r['pmd_ci95'] for r in ssl])*scale
        axis.errorbar(x, y, yerr=[y-limits[:,0], limits[:,1]-y], fmt='o-', capsize=4,
                      label='SSL, seed 42')
    else:
        axis.plot(x, y, 'o-', label='SSL, seed 42')
    reference = next(r for r in sweep if r['supervised'] == 0)
    if key == 'pmd':
        low, high = reference['pmd_ci95']
        axis.errorbar([0], [reference[key]*scale],
            yerr=[[scale*(reference[key]-low)], [scale*(high-reference[key])]],
            fmt='s', color='C1', capsize=4, label='UL reference, seed 10')
    else:
        axis.scatter([0], [reference[key]*scale], marker='s', color='C1', label='UL reference, seed 10')
    axis.set_xticks(range(5), ['0','100','500','1,000','5,000'])
    axis.set(xlabel='Supervised updates (85,000 total)', ylabel=label)
    axis.grid(alpha=.25)
    axis.legend(fontsize=9)
fig.suptitle('Completed supervised-fraction sweep: independent-test evaluation\nThresholds and beams selected before testing; PMD bars are 95% Wilson test-frame intervals', fontsize=13)
for extension in ['png','svg']:
    fig.savefig(output/f'fraction_sweep.{extension}', dpi=180)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
for i, allocation in enumerate([0,100,500,1000,5000]):
    series = sorted([r for r in data if r['supervised'] == allocation], key=lambda r:r['sensing_snr_db'])
    assert len(series) == 5, (allocation, len(series))
    label = 'UL, seed 10' if allocation == 0 else f'SSL: {allocation:,} SL updates, seed 42'
    x = [r['sensing_snr_db'] for r in series]
    for axis, key, ylabel, scale in zip(axes.flat,
        ['pmd','rmse_detected','pfa','ser'],
        ['Misdetection probability [%]','Detected-target position RMSE [m]',
         'False-alarm probability [%]','Symbol error rate [%]'], [100,1,100,100]):
        axis.plot(x, [r[key]*scale for r in series], 'o-', color=f'C{i}', label=label)
        axis.set(xlabel='Test sensing SNR [dB]', ylabel=ylabel, xticks=[5,10,15,20,25])
        axis.grid(alpha=.25)
axes[1,0].axhline(1, color='gray', linestyle='--', linewidth=1, label='Nominal PFA 1%')
axes[1,1].axhline(1, color='gray', linestyle='--', linewidth=1, label='Validation SER limit 1%')
for axis in axes[0]:
    axis.legend(fontsize=8)
axes[1,0].text(5.2, 1.005, 'Nominal PFA 1%', fontsize=9, color='dimgray')
axes[1,1].text(5.2, 1.005, 'Validation SER limit 1%', fontsize=9, color='dimgray')
fig.suptitle('Sensing-SNR transfer of frozen 85,000-update models\nA beam is selected on validation separately at each SNR; communication SNR stays at 20 dB', fontsize=13)
for extension in ['png','svg']:
    fig.savefig(output/f'snr_transfer.{extension}', dpi=180)
plt.close(fig)
print(f'Created fraction_sweep and snr_transfer plots in {output}')
