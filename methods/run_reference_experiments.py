"""Run the full SL and UL references sequentially, with persistent logs."""
from pathlib import Path
import os
import subprocess
import sys
from datetime import datetime

root = Path(__file__).resolve().parents[1]
os.chdir(root)
logs = root / 'results' / 'logs'
logs.mkdir(exist_ok=True)
env = os.environ.copy()
env['MPLBACKEND'] = 'Agg'
env['PYTHONUNBUFFERED'] = '1'
env['MPLCONFIGDIR'] = str(root / '.mplconfig-training')
status = logs / 'reference_status.txt'
for method in ['supervised_learning', 'unsupervised_learning']:
    with status.open('a') as stream:
        stream.write(f'{datetime.now().isoformat()} Starting {method}\n')
    with (logs / f'{method}_85000.log').open('w') as stream:
        result = subprocess.run(
            [sys.executable, '-u', str(root / 'methods' / f'{method}.py')],
            stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=root)
    with status.open('a') as stream:
        stream.write(f'{datetime.now().isoformat()} {method} exit={result.returncode}\n')
    if result.returncode:
        sys.exit(result.returncode)
