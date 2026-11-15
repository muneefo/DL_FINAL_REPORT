"""Independent scene/noise localization test with saved per-scene error tails."""
from pathlib import Path
import argparse
import json
import hashlib
import sys
import numpy as np
import torch
from dataclasses import replace

root = Path(__file__).resolve().parents[1]
sys.path[1:1] = [str(root), str(root/'lib')]
from simulation_parameters import *
from lib.scene_pool import SimConfig, draw_scenes, subset, synthesize, run_receiver

parser = argparse.ArgumentParser()
parser.add_argument('checkpoints', nargs='+')
parser.add_argument('--output', required=True)
parser.add_argument('--samples', type=int, default=5000)
args = parser.parse_args()
output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=True)
for filename in args.checkpoints:
    source = Path(filename).resolve()
    with np.load(source) as z:
        config = json.loads(str(z['config_json'])) if 'config_json' in z else {}
        learned = z['learned_d']
        if learned.ndim == 2 and learned.shape[1] == K and learned.shape[0] != K:
            learned = learned[-1]
        truth = z['true_d'] if 'true_d' in z else ant_d.detach().cpu().numpy()
    coverage = float(config.get('coverage', 60))
    identifier = hashlib.sha256(str(source).encode()).hexdigest()[:12]
    dest = output / f'{source.stem}_{identifier}.npz'
    if dest.exists():
        continue
    d = torch.tensor(learned, dtype=torch.float32, device=device).reshape(K, 1)
    true = torch.tensor(truth, dtype=torch.float32, device=device).reshape(K, 1)
    sim = replace(SimConfig.from_globals(globals()), ant_d=true)
    scenes = draw_scenes(args.samples, sim, -coverage*np.pi/180, coverage*np.pi/180,
        span_min_theta, span_max_theta, range_mean_min, range_mean_max,
        span_min_range, span_max_range, p_present=1., seed=93001)
    torch.manual_seed(94001)
    parts = {key: [] for key in ['true_range', 'est_range', 'true_angle', 'est_angle', 'gain']}
    with torch.no_grad():
        for start in range(0, args.samples, 128):
            sc = subset(scenes, list(range(start, min(start+128, args.samples))))
            Y, symbols, _, A, mask = synthesize(sc, d, sim)
            _, angle, distance = run_receiver(d, sc, sim, Y, symbols, A, mask)
            for key, value in dict(true_range=sc['range'], est_range=distance,
                true_angle=sc['theta'], est_angle=angle, gain=sc['alpha'].abs()).items():
                parts[key].append(value.detach().cpu().numpy().reshape(-1))
    data = {k: np.concatenate(v) for k, v in parts.items()}
    radial = (data['true_range']-data['est_range'])**2
    angular = 2*data['true_range']*data['est_range']*(1-np.cos(data['true_angle']-data['est_angle']))
    e2 = np.maximum(0, radial+angular)
    tail = np.abs(data['true_range']-data['est_range']) > 2.5*float(range_res)
    top = max(1, int(np.ceil(.01*len(e2))))
    summary = dict(source=str(source), coverage=coverage, n=args.samples, scene_seed=93001,
        noise_seed=94001, rmse=float(np.sqrt(e2.mean())), median=float(np.median(np.sqrt(e2))),
        radial_mse_share=float(radial.sum()/e2.sum()), tail_fraction=float(tail.mean()),
        range_tail_mse_share=float(e2[tail].sum()/e2.sum()),
        top_one_percent_mse_share=float(np.sort(e2)[-top:].sum()/e2.sum()),
        median_gain_tail=float(np.median(data['gain'][tail])) if tail.any() else None,
        median_gain_rest=float(np.median(data['gain'][~tail])) if (~tail).any() else None)
    np.savez_compressed(dest, **data, squared_position_error=e2, radial_squared_error=radial,
                        angular_contribution=angular, range_tail=tail, summary_json=json.dumps(summary))
    print(json.dumps(summary), flush=True)
