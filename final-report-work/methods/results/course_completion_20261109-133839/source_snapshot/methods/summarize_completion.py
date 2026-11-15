"""Create a numerical inventory of independent evaluation stages."""
from pathlib import Path
import json
import csv
import sys

cohort = Path(sys.argv[1]).resolve()
rows = []
for stage in ['stage1_validated', 'stage2_sweep_evaluation', 'stage3_snr']:
    for path in sorted((cohort/stage).glob('*.json')):
        item = json.loads(path.read_text())
        if 'result' not in item:
            continue
        result = item['result']
        test = result.get('test', {})
        rows.append(dict(stage=stage, source=item['source'], status=result['status'],
            sensing_snr_db=result.get('parameters', {}).get('snr'),
            pmd=test.get('pmd'), pmd_ci95=json.dumps(test.get('pmd_ci95')),
            pfa=test.get('pfa'), pfa_ci95=json.dumps(test.get('pfa_ci95')),
            ser=test.get('ser'), rmse_detected=test.get('rmse_detected'),
            validation_ser_limit=result.get('validation_ser_limit'),
            test_ser_exceeds_validation_limit=test.get('ser', 0) > .01))
if rows:
    with (cohort/'independent_evaluation.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
lines = ['# Completion-stage evidence', '',
    'Thresholds use independent null calibration; beams are selected on validation data and frozen before testing.',
    'No test result is used to change the operating point. Test SER can exceed the validation constraint and is explicitly reported.',
    'Ranges are sampled per frame, correcting legacy shared-range batches. These results must not overwrite legacy reproduction evidence.',
    'Wilson intervals quantify conditional independent test-frame uncertainty for each frozen model; they do not establish hardware-population significance.', '',
    f'Completed checkpoint/SNR evaluations: {len(rows)}.', '',
    'See independent_evaluation.csv, per-checkpoint JSON metadata, and stage4_localization per-scene arrays.',
    'The course rubric/page limit and final human contribution declaration still require confirmation.']
(cohort/'FINDINGS.md').write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines))
