"""Create nominal and true-spacing controls without optimization."""
from pathlib import Path
import sys
import argparse
import json
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path[1:1] = [str(root), str(root/'lib')]
from simulation_parameters import *
parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
out = Path(args.output).resolve()
out.mkdir(parents=True, exist_ok=True)
arrays = {'reference': ant_d.detach().cpu().numpy()}
study = root/'methods/results/hardware_study_20261108-160301'
for hardware in [3101, 3102, 3103]:
    path = next((study/f'hw{hardware}_th60').glob('active_weak_*_s0_*.npz'))
    with np.load(path) as z:
        arrays[f'hw{hardware}'] = z['true_d'].reshape(K, 1)
for name, truth in arrays.items():
    for kind in ['nominal', 'known']:
        d = np.ones((K, 1), dtype=np.float32)*float(lamb)/2 if kind == 'nominal' else truth.copy()
        config = dict(control=kind, hardware=name, coverage=60,
                      note='no training; known control uses true array in beam and receiver design')
        np.savez(out/f'{kind}_{name}.npz', learned_d=d, true_d=truth, config_json=json.dumps(config))
print(f'Created eight nominal/known controls in {out}')
