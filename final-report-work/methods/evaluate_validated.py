"""Evaluate saved checkpoints with frozen validation-selected operating points."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path[1:1] = [str(root), str(root / 'lib')]
from simulation_parameters import *
from lib.validated_evaluation import ValidatedEvaluator

parser = argparse.ArgumentParser()
parser.add_argument('checkpoints', nargs='+')
parser.add_argument('--output', required=True)
parser.add_argument('--snrs', nargs='+', type=float, default=[15.])
parser.add_argument('--sector', nargs=2, type=float, default=[-40., -20.])
parser.add_argument('--n-cal', type=int, default=10000)
parser.add_argument('--n-validation', type=int, default=10000)
parser.add_argument('--n-test', type=int, default=20000)
parser.add_argument('--beams', type=int, default=4)
args = parser.parse_args()
output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=True)
cache = output / 'cache'
cache.mkdir(exist_ok=True)
for checkpoint in args.checkpoints:
    source = Path(checkpoint).resolve()
    with np.load(source) as saved:
        learned = saved['learned_d']
        if learned.ndim == 2 and learned.shape[1] == K and learned.shape[0] != K:
            learned = learned[-1]
        elif learned.ndim == 3:
            learned = learned[-1]
        truth = saved['true_d'] if 'true_d' in saved else ant_d.detach().cpu().numpy()
        config = str(saved['config_json']) if 'config_json' in saved else '{}'
    assert learned.size == K and truth.size == K, source
    for snr in args.snrs:
        keydata = dict(snr=snr, sector=args.sector, n_cal=args.n_cal,
                       n_validation=args.n_validation, n_test=args.n_test, beams=args.beams)
        h = hashlib.sha256(np.asarray(learned, dtype='float32').tobytes()
                           + np.asarray(truth, dtype='float32').tobytes()
                           + json.dumps(keydata, sort_keys=True).encode()).hexdigest()
        cached = cache / f'{h}.json'
        if cached.exists():
            result = json.loads(cached.read_text())
        else:
            evaluator = ValidatedEvaluator(globals(), truth, learned, snr, args.sector)
            result, sample = evaluator.evaluate(args.n_cal, args.n_validation, args.n_test, args.beams)
            result.update(parameters=keydata)
            cached.write_text(json.dumps(result, indent=2))
            if sample is not None:
                np.savez_compressed(cache / f'{h}_frames.npz', **sample)
        identity = hashlib.sha256(str(source).encode()).hexdigest()[:10]
        dest = output / f'{source.stem}_{identity}_snr{snr:g}.json'
        record = dict(source=str(source), config_json=config, result=result, cache_key=h,
                      frame_data=str(cache / f'{h}_frames.npz'))
        dest.write_text(json.dumps(record, indent=2))
        print(f'{source.name} SNR={snr:g}: {result["status"]}; test={result.get("test")}', flush=True)
