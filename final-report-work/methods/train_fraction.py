"""Fresh, resumable online SSL run with an explicit fixed total-update budget."""
from pathlib import Path
import argparse
import inspect
import json
import os
import sys
import time

root = Path(__file__).resolve().parents[1]
sys.path[1:1] = [str(root), str(root / 'lib')]
from simulation_parameters import *

parser = argparse.ArgumentParser()
parser.add_argument('--supervised', type=int, required=True)
parser.add_argument('--total', type=int, default=85000)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--chunk', type=int, default=1000)
parser.add_argument('--max-chunks', type=int, default=0, help='testing: stop cleanly after this many chunks')
args = parser.parse_args()
assert 0 <= args.supervised <= args.total
out = Path(args.output).resolve()
out.mkdir(parents=True, exist_ok=True)
tag = f'fraction_sl{args.supervised}_total{args.total}_seed{args.seed}'
final = out / f'{tag}.npz'
if final.exists():
    print(f'Already complete: {final}')
    raise SystemExit(0)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
net = PertNet(assumed_d.detach().cpu().clone()).to(device)
sl = torch.optim.Adam(net.parameters(), lr=lr_super)
ul = torch.optim.Adam(net.parameters(), lr=lr_unsuper, maximize=True)
scheduler = torch.optim.lr_scheduler.StepLR(sl, step_size=50000, gamma=.1)
resume = out / f'{tag}_resume.pt'
done, seconds, losses_sl, losses_ul = 0, 0., [], []
fingerprint = dict(seed=args.seed, supervised=args.supervised, total=args.total,
                   batch=batch_size, coverage=theta_mean_max_deg, lr_super=lr_super, lr_unsuper=lr_unsuper)
if resume.exists():
    state = torch.load(resume, map_location=device, weights_only=True)
    if state['fingerprint'] != fingerprint:
        raise ValueError('Resume configuration differs from saved training settings')
    net.load_state_dict(state['network'])
    sl.load_state_dict(state['sl']); ul.load_state_dict(state['ul'])
    scheduler.load_state_dict(state['scheduler'])
    torch.set_rng_state(state['cpu_rng'].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
    done, seconds = state['done'], state['seconds']
    losses_sl, losses_ul = state['loss_sl'], state['loss_ul']
    print(f'Resuming at update {done}', flush=True)

mapping = dict(network=net, theta_span_min=span_min_theta, theta_span_max=span_max_theta,
    range_span_min=span_min_range, range_span_max=span_max_range, numAntennas=K,
    numSubcarriers=S, true_spacing=ant_d, epoch_test_list=[])
base = globals().copy()
base.update(mapping)
signature = inspect.signature(trainNetwork)
chunks = 0
while done < args.total:
    is_sl = done < args.supervised
    end = min(done+args.chunk, args.supervised if is_sl else args.total)
    base.update(init_train_it=done, end_train_it=end, optimizer=sl if is_sl else ul,
                scheduler=scheduler if is_sl else None, SL_flag=is_sl, sch_flag=is_sl)
    kwargs = {name: base[name] for name in signature.parameters if name in base}
    start = time.perf_counter()
    result = trainNetwork(**kwargs)
    seconds += time.perf_counter()-start
    (losses_sl if is_sl else losses_ul).extend(result[0])
    done = end
    state = dict(network=net.state_dict(), sl=sl.state_dict(), ul=ul.state_dict(),
        scheduler=scheduler.state_dict(), cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        done=done, seconds=seconds, loss_sl=losses_sl, loss_ul=losses_ul, fingerprint=fingerprint)
    temporary = resume.with_suffix('.tmp')
    torch.save(state, temporary)
    os.replace(temporary, resume)
    (out / f'{tag}_progress.json').write_text(json.dumps(dict(done=done, total=args.total,
        supervised=args.supervised, seed=args.seed, seconds=seconds, status='training'), indent=2))
    print(f'{tag}: {done}/{args.total}; elapsed training {seconds/3600:.2f} h', flush=True)
    chunks += 1
    if args.max_chunks and chunks >= args.max_chunks and done < args.total:
        print('Stopped at requested test checkpoint; rerun without --max-chunks to resume', flush=True)
        raise SystemExit(0)
config = dict(seed=args.seed, supervised_updates=args.supervised, total_updates=args.total,
    batch=batch_size, coverage=theta_mean_max_deg, sensing_snr_db=float(SNR_sens_dB),
    lr_super=lr_super, lr_unsuper=lr_unsuper, scheduler_step=50000, scheduler_gamma=.1,
    torch_version=torch.__version__, device=device, training_seconds=seconds,
    training_observations=args.total*batch_size, labelled_observations=args.supervised*batch_size,
    data_policy='fresh online scenes each minibatch; observation counts are not queried fixed-pool labels')
np.savez(final, learned_d=net.d.detach().cpu().numpy().copy(),
    true_d=ant_d.detach().cpu().numpy().copy(), loss_super=losses_sl,
    loss_unsuper=losses_ul, config_json=json.dumps(config), completed_updates=done)
(out / f'{tag}_progress.json').write_text(json.dumps(dict(done=done, total=args.total,
    supervised=args.supervised, seed=args.seed, seconds=seconds, status='complete'), indent=2))
print(f'Saved {final}', flush=True)
