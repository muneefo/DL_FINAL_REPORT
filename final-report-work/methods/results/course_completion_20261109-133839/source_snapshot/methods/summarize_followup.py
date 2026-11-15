"""Summarize paired central-sector results without pooling Monte Carlo seeds."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
folder = root / 'methods/results'
output = folder / 'followup_summary'
output.mkdir(exist_ok=True)
rules = ['weak', 'trace', 'full', 'weak_noRC', 'full_noRC', 'random', 'fixed',
         'maxangle', 'diversity', 'uncertainty']
values = {}
report_rows = []
for rule in rules:
    samples = []
    for seed in range(5):
        paths = [p for p in folder.glob(f'isac_central_active_{rule}_s{seed}_th10_cpu_*.npz')
                 if p.stem[-15:] >= '20261108-142922']
        with np.load(sorted(paths)[-1]) as z:
            samples.append(float(z['best_pmd_at_ser_limit']))
    values[rule] = np.array(samples)

with (output / 'central_detection.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['rule', 'n', 'mean_pmd', 'ci95_halfwidth', 'paired_difference_vs_random', 'paired_ci95_halfwidth'])
    for rule, sample in values.items():
        difference = sample - values['random']
        mean, ci = sample.mean(), 2.776 * sample.std(ddof=1) / np.sqrt(5)
        paired_ci = 2.776 * difference.std(ddof=1) / np.sqrt(5)
        writer.writerow([rule, 5, mean, ci, difference.mean(), paired_ci])
        report_rows.append(f'| {rule} | {mean*100:.3f}% | {ci*100:.3f} pp | '
                           f'{difference.mean()*100:.3f} pp | {paired_ci*100:.3f} pp |')
        print(f'{rule}: PMD {mean:.6f} +/- {ci:.6f}; paired vs random {difference.mean():.6f} +/- {paired_ci:.6f}')

fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
means = [values[r].mean() * 100 for r in rules]
ci = [2.776 * values[r].std(ddof=1) / np.sqrt(5) * 100 for r in rules]
ax.bar(rules, means, yerr=ci, capsize=4)
ax.set(ylabel='Misdetection probability [%]',
       title='Central sensing sector [-10, 10] degrees; SER <= 1%\nFive optimization seeds, 95% t intervals; shared evaluation seed')
ax.tick_params(axis='x', rotation=35)
ax.grid(axis='y', alpha=0.25)
fig.savefig(output / 'central_detection.png', dpi=180)
plt.close(fig)

diagnostics = json.loads((folder / 'acquisition_diagnostics_v2.json').read_text())
records = diagnostics['matrix_diagnostics']
max_residual = max(r['gain_collinearity_relative_residual'] for r in records)
report = '''# Active-calibration follow-up findings

The original 50 repaired runs are evaluated without retraining or changing acquisition.
Central sensing sector: [-10,10] degrees, within narrow-training support. Communication
sector remains [30,50] degrees. This is one central sector, not an average over the
random sectors used in training. Transfer evaluation files remain unchanged.

## Acquisition agreement

Across five seeds, weak, trace, D-optimal and both noRC variants have identical
queried labels and every saved model snapshot. Actual candidates were reconstructed
at rounds 1, 5 and 10 for all five seeds; all reconstructed weak queries matched.
Candidate label-information gains are nearly scalar multiples of one common matrix
within each inspected round. Maximum relative collinearity residual: RESIDUAL.
The weak subspace has dimension one in those rounds. Weak and full-trace scores have
correlation essentially one. Thus this experiment does not establish benefits from
weak-subspace targeting or greedy score updates. The synthetic anisotropic test
distinguishes weak and trace, so their implementations are not merely aliases.
These observations describe inspected rounds and this single-target fixed-array
setting; they do not establish equivalence for all models or hardware draws.

## Central-sector detection

Lower PMD is better. 9,984 samples per beam, 4x4 beam grid, shared test seed 54321.
Reported metric: minimum measured PMD among tested beam points with SER <= 1%.
This is a descriptive measured frontier: beam selection and scoring use the same
evaluation data. It is not a separately validated operating-point guarantee.
Intervals are Student-t intervals across five optimization seeds. They do not
measure Monte Carlo uncertainty or generalization across independent hardware draws.
Paired differences use matching optimization seeds and random as the reference.

| Rule | Mean PMD | 95% CI halfwidth | Paired difference vs random | Paired CI halfwidth |
|---|---:|---:|---:|---:|
ROWS

## Next decisions

Use the paired intervals to assess whether acquisition improves over random and
the fixed-label control. Do not attribute any gain to weak-direction targeting:
the tested information-based methods queried the same scenes. Before expanding
the experiment grid, assess multiple independent hardware draws and broader
scenarios where gain directions may differ; report equivalence if it persists.
The labeled-fraction sweep, generalization grid, independent localization test
pool, complete provenance capture and final manuscript remain outstanding.
'''.replace('RESIDUAL', f'{max_residual:.3e}').replace('ROWS', '\n'.join(report_rows))
(output / 'FINDINGS.md').write_text(report, encoding='utf-8')
