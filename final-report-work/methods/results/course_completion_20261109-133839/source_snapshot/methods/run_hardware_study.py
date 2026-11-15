"""Factorial pilot: independent hardware draws x coverage x optimization seeds."""
from pathlib import Path
import os
import sys
import json
import subprocess
import hashlib
from datetime import datetime

root = Path(__file__).resolve().parents[1]
stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
cohort = root / 'methods/results' / f'hardware_study_{stamp}'
cohort.mkdir(parents=True)
manifest = dict(started=stamp, hardware_seeds=[3101, 3102, 3103],
                optimization_seeds=[0, 1, 2], coverage_degrees=[10, 60],
                n_training_runs=126, jobs=[],
                inference_unit='hardware draw; optimizer seeds averaged within hardware',
                test_seed=54321, ser_limit=0.01)
manifest['source_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
    for directory in ['lib', 'methods'] for p in (root / directory).glob('*.py')}
(cohort / 'manifest.json').write_text(json.dumps(manifest, indent=2))


def job(name, script, arguments, env):
    record = dict(name=name, args=arguments, started=datetime.now().isoformat())
    manifest['jobs'].append(record)
    (cohort / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    with (cohort / f'{name}.log').open('w') as stream:
        result = subprocess.run([sys.executable, '-u', str(root / 'methods' / script), *arguments],
                                cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
    record.update(exit_code=result.returncode, finished=datetime.now().isoformat())
    (cohort / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


env = os.environ.copy()
env.update(ISAC_RUN_MODE='cpu', MPLBACKEND='Agg', PYTHONUNBUFFERED='1',
           MPLCONFIGDIR=str(root / '.mplconfig'))
# Record actual interpreter/dependency versions, without changing installed packages.
with (cohort / 'environment.txt').open('w') as stream:
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, stderr=subprocess.STDOUT)
    stream.write(f'\nPython executable: {sys.executable}\nPython: {sys.version}\n')
for coverage in manifest['coverage_degrees']:
    env['ISAC_COVERAGE'] = str(coverage)
    for hardware in manifest['hardware_seeds']:
        name = f'hw{hardware}_th{coverage}'
        folder = cohort / name
        folder.mkdir()
        env.update(ISAC_HARDWARE_SEED=str(hardware), ISAC_RESULTS_DIR=str(folder))
        job(name + '_matched', 'active_calibration.py',
            ['--rules', 'weak', 'trace', 'full', 'random', 'fixed', '--seeds', '0', '1', '2'], env)
        job(name + '_noRC', 'active_calibration.py',
            ['--rules', 'weak', 'full', '--no-redundancy-control', '--seeds', '0', '1', '2'], env)
        checkpoints = sorted(folder.glob('active_*.npz'))
        assert len(checkpoints) == 21
        job(name + '_isac', 'evaluate_active_isac.py', [str(p) for p in checkpoints], env)
job('summary', 'summarize_hardware_study.py', [str(cohort)], env)
manifest['finished'] = datetime.now().isoformat()
(cohort / 'manifest.json').write_text(json.dumps(manifest, indent=2))
