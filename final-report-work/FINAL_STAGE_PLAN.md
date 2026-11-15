# Final report experiment status

## Completed before this working copy

The parent workspace contains SL, UL and sequential SSL at 85,000 updates.
Those reproduction runs use seed 42 in the current parent configuration.
The midterm reproduction used seed 10; avoid interpreting differences between
those cohorts as the effect of training duration alone.

## Recovered source

Source: https://github.com/andrewliang-uwcs/867-deep-learning
This isolated checkout preserves the original parent workspace and its results.
Historical active outputs remain available, but protocol-v2 results must be
summarized separately using their cohort start timestamp.

## Protocol-v2 repairs and controls

- Clone each spacing snapshot so rounds retain independent parameter values.
- Restore CPU/CUDA/NumPy random states around acquisition and validation.
- Same initial labels, training minibatch generator, and update counts per seed.
- Fixed-label control: retain initial labels, execute every training round.
- Trace control: same information matrices and gauge removal as weak selection,
  retaining all directions; D-optimal `full` remains a separate comparator.
- Existing noRC switch isolates greedy score updates.
- Save initial labels, configuration, software version, timing and protocol ID.
- Environment overrides: ISAC_RUN_MODE and ISAC_COVERAGE.

## Current experiment queue

Five seeds (0–4), narrow coverage (10 degrees), existing pilot-sized settings:
eight rules (weak, trace, full, random, fixed, maxangle, diversity, uncertainty),
then weak/full noRC: 50 runs in total. `cpu` names the pilot preset; computation
uses CUDA when available. Summary includes paired differences and 95% intervals.
Fixed-label comparisons use training rounds on the horizontal axis.
Logs and manifest: methods/results/final_stage_<start timestamp>/.

After training, evaluate all final checkpoints with independent seed 54321,
9,984 test samples and a 4x4 ISAC beam grid (pilot precision).
The prespecified summary is minimum measured PMD among beam points with
SER <= 0.01. Missing feasible points are recorded as NaN, never extrapolated.
The original sensing/communication sectors are used. For narrow-coverage
training this is a transfer test; in-distribution detection remains to be added.
Smoke runs only test execution; they are not research evidence.

## Still to complete

1. Completed: corrected five-seed cohort, 50 transfer ISAC evaluations, and
   acquisition diagnostics. The calibration paired interval includes zero.
   Weak, trace, D-optimal and noRC variants have identical queries and models;
   reconstructed candidates in 15 rounds have nearly collinear label gains.
2. Completed: all 50 central-sector [-10,10] sensing evaluations and summary.
   This sector lies within training support but does not reproduce the full
   random-sector training distribution. See followup_<timestamp>/status.json.
   Add Monte Carlo uncertainty and validate a fixed operating point separately.
   Guided mean PMD is 8.7287%, random 8.7362%, fixed 8.7400% at measured
   SER <= 1%; the guided/random paired interval includes zero. This provides
   no clear central-sector detection advantage. Findings and figures are in
   methods/results/followup_summary/.
3. Labeled-fraction sweep at fixed total updates. Distinguish unique acquired
   labels from supervised update counts and reused training scenes.
4. Broad coverage and SNR generalization experiments; distinguish independent
   hardware impairment draws from optimization seeds on one fixed array.
   Hardware/coverage study launched in hardware_study_20261108-160301:
   3 hardware seeds (3101–3103), 2 coverages (10 and 60 degrees), 3 optimization
   seeds (0–2), 7 variants = 126 training runs, each followed by common-sector
   ISAC evaluation. Hardware generation uses a private random generator and
   the reference lambda/25 Gaussian spacing law with ordered-position rejection.
   Tests confirmed reproducibility, distinct arrays and unchanged training RNG.
   Narrow and broad scene pools are shared across hardware draws and methods.
   Summary treats hardware draws as independent units, averaging optimization
   seeds within each hardware. SNR variation remains outstanding.
5. Independent held-out scene pools for localization and per-scene tail analysis.
6. Freeze full environment/config/source provenance and verify clean rerun.
7. Update final manuscript, contribution declaration and presentation.

Do not launch the full labeled-fraction/SNR/impairment grid before reviewing
pilot controls: it would consume substantial compute without resolving the
main acquisition question first.
