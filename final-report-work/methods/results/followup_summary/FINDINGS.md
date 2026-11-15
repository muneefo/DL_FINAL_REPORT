# Active-calibration follow-up findings

The original 50 repaired runs are evaluated without retraining or changing acquisition.
Central sensing sector: [-10,10] degrees, within narrow-training support. Communication
sector remains [30,50] degrees. This is one central sector, not an average over the
random sectors used in training. Transfer evaluation files remain unchanged.

## Acquisition agreement

Across five seeds, weak, trace, D-optimal and both noRC variants have identical
queried labels and every saved model snapshot. Actual candidates were reconstructed
at rounds 1, 5 and 10 for all five seeds; all reconstructed weak queries matched.
Candidate label-information gains are nearly scalar multiples of one common matrix
within each inspected round. Maximum relative collinearity residual: 3.511e-09.
The weak subspace has dimension one in those rounds. Weak and full-trace scores have
correlation essentially one. Thus this experiment does not establish benefits from
weak-subspace targeting or greedy score updates. The synthetic anisotropic test
distinguishes weak and trace, so their implementations are not merely aliases.
These observations describe inspected rounds and this single-target fixed-array
setting; they do not establish equivalence for all models or hardware draws.

## Central-sector detection

Lower PMD is better. 9,984 samples per beam, 4x4 beam grid, shared test seed 54321.
Reported metric: minimum measured PMD among tested beam points with SER <= 1%.
This is a descriptive measured frontier: beam selection and scoring use the same
evaluation data. It is not a separately validated operating-point guarantee.
Intervals are Student-t intervals across five optimization seeds. They do not
measure Monte Carlo uncertainty or generalization across independent hardware draws.
Paired differences use matching optimization seeds and random as the reference.

| Rule | Mean PMD | 95% CI halfwidth | Paired difference vs random | Paired CI halfwidth |
|---|---:|---:|---:|---:|
| weak | 8.729% | 0.080 pp | -0.008 pp | 0.111 pp |
| trace | 8.729% | 0.080 pp | -0.008 pp | 0.111 pp |
| full | 8.729% | 0.080 pp | -0.008 pp | 0.111 pp |
| weak_noRC | 8.729% | 0.080 pp | -0.008 pp | 0.111 pp |
| full_noRC | 8.729% | 0.080 pp | -0.008 pp | 0.111 pp |
| random | 8.736% | 0.075 pp | 0.000 pp | 0.000 pp |
| fixed | 8.740% | 0.054 pp | 0.004 pp | 0.111 pp |
| maxangle | 8.807% | 0.097 pp | 0.071 pp | 0.086 pp |
| diversity | 8.796% | 0.203 pp | 0.060 pp | 0.222 pp |
| uncertainty | 10.453% | 4.247 pp | 1.717 pp | 4.230 pp |

## Next decisions

Use the paired intervals to assess whether acquisition improves over random and
the fixed-label control. Do not attribute any gain to weak-direction targeting:
the tested information-based methods queried the same scenes. Before expanding
the experiment grid, assess multiple independent hardware draws and broader
scenarios where gain directions may differ; report equivalence if it persists.
The labeled-fraction sweep, generalization grid, independent localization test
pool, complete provenance capture and final manuscript remain outstanding.
