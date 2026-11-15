"""Check the repaired acquisition rules and saved smoke-run invariants."""
from pathlib import Path
import sys
import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from lib.identifiability import select_batch

# Weak-direction and full-trace criteria must differ on anisotropic information.
F = torch.diag(torch.tensor([0.001, 100.], dtype=torch.float64))
gains = torch.stack([torch.diag(torch.tensor([2., 0.], dtype=torch.float64)),
                     torch.diag(torch.tensor([0., 20.], dtype=torch.float64))])
Q = torch.eye(2, dtype=torch.float64)
assert select_batch(F, gains, [0, 1], 1, Q, rule='weak') == [0]
assert select_batch(F, gains, [0, 1], 1, Q, rule='trace') == [1]

runs = []
for rule in ['weak', 'trace', 'random', 'fixed']:
    paths = sorted((root / 'methods/results').glob(f'active_{rule}_s42_th10_smoke_*.npz'))
    with np.load(paths[-1]) as z:
        runs.append({k: z[k] for k in z.files})
for run in runs:
    assert run['protocol_version'] == 2
    assert len(run['learned_d']) == 3
    assert not np.array_equal(run['learned_d'][0], run['learned_d'][-1])
    np.testing.assert_array_equal(run['initial_labels'], runs[0]['initial_labels'])
    np.testing.assert_array_equal(run['learned_d'][0], runs[0]['learned_d'][0])
    assert np.isfinite(run['rmse_pos']).all()
assert runs[-1]['n_labels'].tolist() == [4, 4, 4]
assert runs[-1]['chosen'].size == 0
print('PASS: criteria, independent snapshots, shared initial labels/model, fixed-label update budget')
