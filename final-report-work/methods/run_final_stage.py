"""Run repaired multi-seed active controls; retain logs and a cohort manifest."""
from pathlib import Path
import subprocess
import sys
import os
import json
from datetime import datetime

root = Path(__file__).resolve().parents[1]
stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
logs = root / 'methods/results' / f'final_stage_{stamp}'
logs.mkdir(parents=True)
env = os.environ.copy()
env.update(ISAC_RUN_MODE='cpu', ISAC_COVERAGE='10', MPLBACKEND='Agg',
           PYTHONUNBUFFERED='1', MPLCONFIGDIR=str(root / '.mplconfig'))
manifest = dict(protocol_version=2, started=stamp, coverage=10,
                mode='cpu', seeds=[0, 1, 2, 3, 4], jobs=[])


def job(name, script, arguments):
    record = dict(name=name, started=datetime.now().isoformat(), args=arguments)
    manifest['jobs'].append(record)
    (logs / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    with (logs / f'{name}.log').open('w') as stream:
        result = subprocess.run([sys.executable, '-u', str(root / 'methods' / script),
                                 *arguments], cwd=root, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    record.update(exit_code=result.returncode, finished=datetime.now().isoformat())
    (logs / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


job('matched_controls', 'active_calibration.py',
    ['--rules', 'weak', 'trace', 'full', 'random', 'fixed', 'maxangle', 'diversity',
     'uncertainty', '--seeds', '0', '1', '2', '3', '4'])
job('no_redundancy', 'active_calibration.py',
    ['--rules', 'weak', 'full', '--no-redundancy-control', '--seeds', '0', '1', '2', '3', '4'])
job('summary', 'summarize_active.py', ['--mode', 'cpu', '--th', '10', '--since', stamp])
checkpoints = [str(p) for p in sorted((root / 'methods/results').glob('active_*_th10_cpu_*.npz'))
               if p.stem[-15:] >= stamp]
job('independent_isac', 'evaluate_active_isac.py', checkpoints)
manifest['finished'] = datetime.now().isoformat()
(logs / 'manifest.json').write_text(json.dumps(manifest, indent=2))
