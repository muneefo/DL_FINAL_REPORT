"""Evaluate final active checkpoints with shared, independent ISAC test randomness.

This uses the original test sectors (-40..-20 sensing, 30..50 communication),
so th10 training is evaluated outside its training coverage. It is an explicit
transfer test, not an in-distribution active-label performance claim.
"""
from pathlib import Path
import argparse
import sys
import json

root = Path(__file__).resolve().parents[1]
sys.path[1:1] = [str(root), str(root / 'lib')]
from simulation_parameters import *

parser = argparse.ArgumentParser()
parser.add_argument('checkpoints', nargs='+')
parser.add_argument('--seed', type=int, default=54321)
parser.add_argument('--ser-limit', type=float, default=0.01)
parser.add_argument('--scenario', choices=['transfer', 'central'], default='transfer',
                    help='central: sensing sector [-10,10] within th10 training support; communication unchanged')
args = parser.parse_args()
test_sector = [-40, -20] if args.scenario == 'transfer' else [-10, 10]
sens_min, sens_max = [torch.tensor(v * np.pi / 180, dtype=torch.float32, device=device)
                      for v in test_sector]
for filename in args.checkpoints:
    source = Path(filename).resolve()
    with np.load(source) as saved:
        if int(saved['protocol_version']) != 2:
            raise ValueError('Only repaired protocol-v2 checkpoints are supported')
        config = json.loads(str(saved['config_json']))
        if args.scenario == 'central' and config['coverage'] != 10:
            raise ValueError('Central scenario is specified for the th10 cohort')
        if config['K'] != K or config['S'] != S:
            raise ValueError('Checkpoint/system dimensions differ')
        d = torch.tensor(saved['learned_d'][-1], dtype=torch.float32, device=device).reshape(K, 1)
        if not np.allclose(saved['true_d'], ant_d.detach().cpu().numpy().reshape(-1)):
            raise ValueError('Checkpoint impairments differ from evaluation configuration')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pd, pfa, angle, rng, pos, ser = testNetworkISAC(
        sigma_sens, sigma_vector_comm, sens_min, sens_max,
        theta_min_comm_test, theta_max_comm_test, range_min_sens_test,
        range_max_sens_test, Ngrid_angle, range_grid, pixels_angle, pixels_range,
        K, S, N0, Delta_f, lamb, ant_d, d, refConst, rho, phi, target_pfa,
        delta_pfa, thresholds_pfa, batch_size, nTestSamples, device)
    pd, ser = np.asarray(pd), np.asarray(ser)
    feasible = ser <= args.ser_limit
    best = float(np.min(1-pd[feasible])) if feasible.any() else float('nan')
    dest = source.with_name(('isac_' if args.scenario == 'transfer' else 'isac_central_') + source.name)
    np.savez(dest, pd_isac=pd, pfa_isac=pfa, rmse_angle_isac=angle,
             rmse_range_isac=rng, rmse_pos_isac=pos, ser_isac=ser,
             source=str(source), test_seed=args.seed, n_test=nTestSamples,
             ser_limit=args.ser_limit, best_pmd_at_ser_limit=best,
             test_sensing_sector_deg=test_sector, test_comm_sector_deg=[30, 50],
             scenario=args.scenario,
             training_config_json=json.dumps(config), evaluation_mode=run_mode)
    print(f'Saved {dest}; PMD at SER <= {args.ser_limit}: {best}')
