"""Summarize robustness using hardware draws as the independent units."""
from pathlib import Path
import sys
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cohort = Path(sys.argv[1]).resolve()
manifest = json.loads((cohort / 'manifest.json').read_text())
rules = ['weak', 'trace', 'full', 'random', 'fixed', 'weak_noRC', 'full_noRC']
hardware_seeds = manifest['hardware_seeds']
optimization_seeds = manifest['optimization_seeds']
rows, agreement, collinearity = [], [], []


def load(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


for coverage in manifest['coverage_degrees']:
    results = {}
    for hardware in hardware_seeds:
        folder = cohort / f'hw{hardware}_th{coverage}'
        for seed in optimization_seeds:
            runs = {}
            for rule in rules:
                paths = list(folder.glob(f'active_{rule}_hw{hardware}_s{seed}_th{coverage}_cpu_*.npz'))
                assert len(paths) == 1, (rule, hardware, seed, paths)
                run = load(paths[0])
                config = json.loads(str(run['config_json']))
                assert config['hardware_seed'] == hardware and config['coverage'] == coverage
                evaluated = load(paths[0].with_name('isac_' + paths[0].name))
                results[(hardware, seed, rule)] = (float(run['calib_err'][-1]),
                    float(evaluated['best_pmd_at_ser_limit']))
                runs[rule] = run
                finite = run['gain_collinearity_residual'][np.isfinite(run['gain_collinearity_residual'])]
                if finite.size:
                    collinearity.append(dict(coverage=coverage, hardware=hardware,
                        seed=seed, rule=rule, max_gain_residual=float(finite.max())))
            for rule in ['trace', 'full', 'weak_noRC', 'full_noRC']:
                agreement.append(dict(coverage=coverage, hardware=hardware, seed=seed, rule=rule,
                    identical_queries=bool(np.array_equal(runs['weak']['chosen'], runs[rule]['chosen'])),
                    identical_models=bool(np.array_equal(runs['weak']['learned_d'], runs[rule]['learned_d']))))
    for metric_index, metric in enumerate(['calibration_error', 'pmd_at_ser_1pct']):
        # First average repeated optimizer seeds within each hardware draw.
        hardware_means = {rule: np.array([np.mean([results[(hw, seed, rule)][metric_index]
            for seed in optimization_seeds]) for hw in hardware_seeds]) for rule in rules}
        for rule in rules:
            sample = hardware_means[rule]
            delta = sample - hardware_means['random']
            ci = 4.303 * sample.std(ddof=1) / np.sqrt(3)
            delta_ci = 4.303 * delta.std(ddof=1) / np.sqrt(3)
            rows.append(dict(coverage=coverage, metric=metric, rule=rule, n_hardware=3,
                n_finite_hardware=int(np.isfinite(sample).sum()),
                optimizer_seeds_per_hardware=3, mean=float(sample.mean()), ci95_halfwidth=float(ci),
                paired_vs_random=float(delta.mean()), paired_ci95_halfwidth=float(delta_ci)))
with (cohort / 'summary.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(cohort / 'acquisition_agreement.json').write_text(json.dumps(
    dict(agreement=agreement, gain_collinearity=collinearity), indent=2))
fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
for axis, coverage in zip(axes, manifest['coverage_degrees']):
    selected = [r for r in rows if r['coverage'] == coverage and r['metric'] == 'pmd_at_ser_1pct']
    axis.bar([r['rule'] for r in selected], [r['mean']*100 for r in selected],
             yerr=np.array([r['ci95_halfwidth']*100 for r in selected]), capsize=3)
    axis.set(title=f'Training sector centres within ±{coverage}°', ylabel='Misdetection probability [%]')
    axis.tick_params(axis='x', rotation=40)
    axis.grid(axis='y', alpha=0.25)
fig.suptitle('Three independent arrays; three optimizer seeds per array\n95% t intervals across array means; SER <= 1%')
fig.savefig(cohort / 'hardware_detection.png', dpi=180)
plt.close(fig)
lines = ['# Hardware and coverage robustness study', '',
    '126 matched training runs: three hardware draws, two coverages, three optimizer seeds, seven variants.',
    'Random spacing uses the reference sorted Gaussian draw with std=lambda/25 and rejects unordered element positions.',
    'Intervals use three hardware means (averaging optimizer seeds within each hardware). With only three independent arrays, uncertainty is substantial.',
    'Common sensing sector [-40,-20] degrees and communication sector [30,50] degrees: outside narrow training support, within broad support.',
    'Detection is the minimum measured PMD on the 4x4 beam grid at measured SER <=1%; 9,984 samples/beam, shared test seed 54321.',
    'This measured frontier does not establish a separately validated operating-point guarantee or Monte Carlo confidence interval.', '',
    'If no beam satisfies the SER constraint, PMD is NaN. No missing hardware draw is dropped or imputed; group means/intervals remain undefined, and summary.csv records finite hardware counts.', '',
    '| Coverage | Metric | Rule | Mean | 95% halfwidth | Paired difference vs random | Paired halfwidth |',
    '|---|---|---|---:|---:|---:|---:|']
for row in rows:
    lines.append(f"| {row['coverage']} | {row['metric']} | {row['rule']} | {row['mean']:.6f} | {row['ci95_halfwidth']:.6f} | {row['paired_vs_random']:.6f} | {row['paired_ci95_halfwidth']:.6f} |")
lines += ['', f"Identical-query comparisons versus weak: {sum(a['identical_queries'] for a in agreement)}/{len(agreement)}.",
          'Inspect acquisition_agreement.json for the per-array results and matrix diagnostics.',
          'No acquisition advantage is established when the paired interval includes zero. Do not attribute benefits to weak-direction targeting when queried labels coincide.']
(cohort / 'FINDINGS.md').write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines))
