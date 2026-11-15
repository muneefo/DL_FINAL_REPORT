"""Format saved numerical evidence into submission tables; no retraining."""
from pathlib import Path
import csv
import json
import argparse

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--cohort', type=Path, default=root/'final-report-work/methods/results/course_completion_20261109-133839')
args = parser.parse_args()
out = root/'reports/submission_tables'
out.mkdir(parents=True, exist_ok=True)
results = root/'final-report-work/methods/results'
calibration = {r['rule']: r for r in csv.DictReader((results/'summary_th10_cpu/summary.csv').open())}
central = {r['rule']: r for r in csv.DictReader((results/'followup_summary/central_detection.csv').open())}
cal_mean_key = next(k for k in calibration['weak'] if k.startswith('final calib err') and k.endswith('mean'))
cal_ci_key = next(k for k in calibration['weak'] if k.startswith('final calib err') and k.endswith('ci95'))
ablation = [r'\begin{table}[t]\centering\small',
    r'\caption{Controlled ablations on the corrected fixed-array cohort. Every variant receives 2,200 updates. Entries are means with 95\% Student-$t$ halfwidths across five optimization seeds. Central-sector misdetection uses the legacy measured-frontier protocol, not the later independent-test evaluator.}\label{tab:final-ablation}',
    r'\setlength{\tabcolsep}{3pt}',
    r'\begin{tabular}{lrrr}\toprule Change & Labels & $e_d$ & $P_{\rm md}$ (\%)\\\midrule']
rows = []
for rule, label, nlabels in [('trace', 'No weak projection (trace)', 60),
                            ('weak_noRC', 'No greedy score update', 60),
                            ('fixed', 'No new-label acquisition', 10),
                            ('weak', 'Proposed weak + greedy', 60)]:
    cm, ci = float(calibration[rule][cal_mean_key]), float(calibration[rule][cal_ci_key])
    pm, pi = float(central[rule]['mean_pmd'])*100, float(central[rule]['ci95_halfwidth'])*100
    ablation.append(f'{label} & {nlabels} & ${cm:.4f}\\pm{ci:.4f}$ & ${pm:.3f}\\pm{pi:.3f}$ ' + r'\\')
    rows.append(dict(rule=rule, change=label, labels=nlabels, calibration_mean=cm,
                     calibration_ci_halfwidth=ci, pmd_percent_mean=pm, pmd_percent_ci_halfwidth=pi))
ablation += [r'\bottomrule\end{tabular}', r'\end{table}']
(out/'ablation_results.tex').write_text('\n'.join(ablation), encoding='utf-8')
with (out/'ablation_results.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

records = {}
for path in (args.cohort/'stage1_validated').glob('*.json'):
    record = json.loads(path.read_text())
    if 'source' in record:
        records[Path(record['source']).name] = record['result']
main = [r'\begin{table}[t]\centering\small',
    r'\caption{Independent-test evaluation of saved 85,000-update models. Thresholds and beams are frozen before the 20,000-frame test. PMD intervals are Wilson test-frame intervals, conditional on each fixed model. These are single trained models, not repeated-seed averages.}\label{tab:independent-main}',
    r'\setlength{\tabcolsep}{3pt}',
    r'\begin{tabular}{lrrrr}\toprule Model & $P_{\rm md}$ [95\% CI] & $P_{\rm fa}$ & SER & RMSE (m)\\\midrule']
main_rows = []
for name, filename in [('SL','SL_85000.npz'), ('UL','UL_85000.npz'), ('SSL','SSL_SL_UL_1000.npz')]:
    if filename not in records:
        raise RuntimeError(f'Independent evaluation not complete for {filename}')
    result = records[filename]
    if result['status'] != 'ok':
        main.append(f'{name} & \\multicolumn{{4}}{{c}}{{No validation-feasible beam}} ' + r'\\')
        continue
    m = result['test']; lo, hi = m['pmd_ci95']
    main.append(f"{name} & ${m['pmd']:.4f}$ [$ {lo:.4f}, {hi:.4f} $] & {m['pfa']:.4f} & {m['ser']:.4f} & {m['rmse_detected']:.3f} " + r'\\')
    main_rows.append(dict(model=name, **{k:m[k] for k in ['pmd','pfa','ser','rmse_detected']}, pmd_lower=lo,pmd_upper=hi))
main += [r'\bottomrule\end{tabular}', r'\end{table}']
(out/'main_results.tex').write_text('\n'.join(main), encoding='utf-8')
if main_rows:
    with (out/'main_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(main_rows[0])); writer.writeheader(); writer.writerows(main_rows)
print(f'Created numerical tables in {out}')
