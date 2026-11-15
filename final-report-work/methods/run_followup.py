"""Run the acquisition diagnosis and central-sector ISAC follow-up."""
from pathlib import Path
import subprocess
import sys
import os
import json
from datetime import datetime

root = Path(__file__).resolve().parents[1]
logs = root / 'methods/results' / ('followup_' + datetime.now().strftime('%Y%m%d-%H%M%S'))
logs.mkdir(parents=True)
env = os.environ.copy()
env.update(ISAC_RUN_MODE='cpu', ISAC_COVERAGE='10', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
status = []
files = [str(p) for p in sorted((root / 'methods/results').glob('active_*_th10_cpu_*.npz'))
         if p.stem[-15:] >= '20261108-142922']
assert len(files) == 50, f'Expected original 50-run cohort, got {len(files)}'
for name, script, arguments in [
    ('diagnostics', 'diagnose_acquisition.py', []),
    ('central_isac', 'evaluate_active_isac.py', ['--scenario', 'central', *files]),
    ('summary', 'summarize_followup.py', []),
]:
    row = dict(job=name, started=datetime.now().isoformat())
    status.append(row)
    (logs / 'status.json').write_text(json.dumps(status, indent=2))
    with (logs / (name + '.log')).open('w') as stream:
        result = subprocess.run([sys.executable, '-u', str(root / 'methods' / script), *arguments],
                                cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
    row.update(exit_code=result.returncode, finished=datetime.now().isoformat())
    (logs / 'status.json').write_text(json.dumps(status, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)
