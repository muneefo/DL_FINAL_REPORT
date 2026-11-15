"""Check calibration and independent evaluation before production runs."""
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path[1:1] = [str(root), str(root / 'lib')]
from simulation_parameters import *
from lib.validated_evaluation import ValidatedEvaluator, null_threshold, wilson

assert null_threshold(np.arange(100.), .1) == 90.
assert np.isinf(null_threshold(np.arange(10.), .01))
assert wilson(0, 100)[0] == 0.
evaluator = ValidatedEvaluator(globals(), ant_d, assumed_d)
a = evaluator.collect(64, 101, .5, 0.)
b = evaluator.collect(64, 102, .5, 0.)
assert len(np.unique(a['true_range'])) == 64, 'Ranges must be independent across frames'
assert not np.array_equal(a['metric'], b['metric'])
c = evaluator.collect(64, 101, .5, 0.)
np.testing.assert_array_equal(a['metric'], c['metric'])
first = evaluator.collect(64, 103, 0., 0., null_only=True)
last = evaluator.collect(64, 103, 1., 1., null_only=True)
np.testing.assert_array_equal(first['metric'], last['metric'])
result, sample = evaluator.evaluate(n_cal=256, n_validation=256, n_test=512, beams=3, ser_limit=1.)
assert result['status'] == 'ok' and result['n_test'] == 512
assert len({result['calibration_seed'], result['validation_seed'], result['test_seed']}) == 3
assert sample['metric'].size == 512 and 0 <= result['test']['pfa'] <= 1
print('PASS: order-statistic calibration; split seeds; framewise ranges; repeatability; null-beam invariance; frozen beam/threshold test')
