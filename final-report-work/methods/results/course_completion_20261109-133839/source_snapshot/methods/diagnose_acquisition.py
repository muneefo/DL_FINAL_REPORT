"""Reconstruct real candidate matrices to explain acquisition agreement.

No training or criterion tuning: inspect saved models and the original seeded
candidate sets. Check the complete recorded query/model equality separately.
"""
from pathlib import Path
import sys
import json
import argparse

root = Path(__file__).resolve().parents[1]
sys.path[1:1] = [str(root), str(root / 'lib')]
from simulation_parameters import *
from lib.scene_pool import SimConfig, load_scenes, subset, num_scenes
from lib.identifiability import (ObsModel, translation_direction, complement_basis,
    pool_information, weak_directions, score_weak, score_full_information, select_batch)

parser = argparse.ArgumentParser()
parser.add_argument('--since', default='20261108-142922')
args = parser.parse_args()
folder = root / 'methods/results'
sim = SimConfig.from_globals(globals())
m = ObsModel(sim)
Q = complement_basis(translation_direction(m).reshape(-1, 1), K)
pool, _ = load_scenes(str(root / 'methods/data/scenes_cpu_th10_train.pt'), device)
records, equalities = [], []
for seed in range(5):
    def load(rule):
        paths = [p for p in folder.glob(f'active_{rule}_s{seed}_th10_cpu_*.npz')
                 if p.stem[-15:] >= args.since]
        with np.load(sorted(paths)[-1]) as z:
            return {k: z[k] for k in z.files}
    base = load('weak')
    settings = json.loads(str(base['config_json']))['settings']
    for rule in ['trace', 'full', 'weak_noRC', 'full_noRC']:
        other = load(rule)
        equalities.append(dict(seed=seed, rule=rule,
            identical_queries=bool(np.array_equal(base['chosen'], other['chosen'])),
            identical_models=bool(np.array_equal(base['learned_d'], other['learned_d']))))
    for rnd in [1, 5, 10]:
        labelled = base['initial_labels'].tolist() + base['chosen'][base['chosen_round'] < rnd].tolist()
        unlabelled = [i for i in range(num_scenes(pool)) if i not in set(labelled)]
        generator = torch.Generator().manual_seed(seed * 1000 + rnd)
        cand = [unlabelled[j] for j in torch.randperm(len(unlabelled), generator=generator)
                [:settings['n_cand']].tolist()]
        net = PertNet(torch.tensor(base['learned_d'][rnd-1], dtype=torch.float32).reshape(K, 1)).to(device)
        torch.manual_seed(1000000 + seed * 1000 + rnd)
        FU, gains, _ = pool_information(net, subset(pool, labelled + cand), sim, m, batch_size=64)
        nl = len(labelled)
        F = (FU[:nl] + gains[:nl]).sum(0) + FU[nl:].sum(0)
        gains = gains[nl:]
        V, eigenvalues = weak_directions(F, Q, settings['tau'])
        weak = score_weak(gains, V)
        trace = score_weak(gains, Q)
        full = score_full_information(gains, F, Q)
        # How well do candidate gains lie on one shared matrix direction?
        flat = gains.reshape(len(cand), -1)
        direction = flat[torch.argmax(torch.linalg.vector_norm(flat, dim=1))]
        coefficients = flat @ direction / (direction @ direction)
        residual = torch.linalg.vector_norm(flat - coefficients[:, None] * direction) / torch.linalg.vector_norm(flat)
        weak_picks = select_batch(F, gains, list(range(len(cand))), 5, Q, rule='weak')
        reconstructed = [cand[i] for i in weak_picks]
        recorded = base['chosen'][base['chosen_round'] == rnd].tolist()
        records.append(dict(seed=seed, round=rnd, weak_dimensions=V.shape[1],
            gauge_free_dimensions=Q.shape[1], eigenvalue_ratio=float(eigenvalues[0]/eigenvalues[-1]),
            gain_collinearity_relative_residual=float(residual),
            weak_trace_max_relative_difference=float((weak-trace).abs().max()/trace.abs().max()),
            weak_trace_correlation=float(np.corrcoef(weak.numpy(), trace.numpy())[0, 1]),
            trace_dopt_correlation=float(np.corrcoef(trace.numpy(), full.numpy())[0, 1]),
            reconstructed_queries_match=reconstructed == recorded))
        print(records[-1], flush=True)
out = folder / 'acquisition_diagnostics_v2.json'
out.write_text(json.dumps(dict(checkpoint_equalities=equalities, matrix_diagnostics=records), indent=2))
assert all(r['reconstructed_queries_match'] for r in records), 'Reconstruction mismatch; investigate before interpreting'
print(f'Saved {out}')
