# Evaluation audit and completion sequence

The legacy code and all saved reproduction results are preserved.

## Findings from source inspection

1. `testNetworkISAC` calibrates detection thresholds on the observations it
   reports. It then interpolates independently sorted PFA, PD and RMSE arrays.
   Sorting the ordinate independently discards its pairing with the abscissa;
   this is particularly unsafe for nonmonotonic RMSE values. The new evaluator
   uses no interpolation.
2. `sensingChannel` repeats scalar angular limits to the batch size but does
   not repeat scalar range limits. Calling it with scalar range limits draws
   one range and broadcasts it over a batch. The new evaluator explicitly
   passes range-limit vectors so every test frame draws an independent range.
3. Choosing the best beam using reported test PMD is descriptive frontier
   estimation, not a validation-selected operating point. The new evaluator
   selects a beam using validation observations and freezes it before testing.

## Replacement protocol

- 10,000 absent-target calibration frames, seed 87001.
- A finite-sample null order statistic sets one threshold at nominal PFA 1%.
  Under the current additive-noise null, receiver null metrics do not depend
  on transmit beam; a smoke test verifies that invariance.
- 10,000 separate validation frames, seed 88001, across a 4x4 beam grid.
- Among beams with validation SER <=1%, choose lowest validation PMD.
- Freeze that beam and threshold, then test 20,000 frames with seed 89001.
- If validation has no feasible beam, record that status without test-driven
  fallback. A test SER violation is reported without reselecting a beam.
- Report Wilson intervals for test-frame PMD/PFA, conditional on the frozen
  model/threshold. SER standard errors use frames as clusters, because symbols
  within a frame share a fading channel. A nominal PFA is not an exact test guarantee.

Passed checks: null threshold order statistic, disjoint split seeds, deterministic
replay, per-frame ranges, null beam invariance, frozen operating-point execution,
and interrupted/resumed training matching an uninterrupted smoke run.

## Sequential completion queue

1. Reassess 176 active checkpoints, three parent reproduction models and eight
   nominal/known-array controls. Identical models reuse explicitly keyed test
   outputs rather than being represented as additional independent evidence.
2. Fresh fixed-budget sweep: 85,000 updates for supervised allocations
   0, 100, 500, 1,000, 5,000, 10,000 and 85,000, each at seeds 10 and 42.
   Batch size stays 3,000; SL learning rate decays at SL update 50,000.
   All fresh runs are needed for consistent provenance; old archives are not
   silently relabeled as matched-seed replicates. Resume state is saved every
   1,000 updates. This stage is substantially longer than the active pilots.
3. Sensing-SNR transfer tests at 5, 10, 20 and 25 dB using frozen models.
4. Independent localization pools: seed 93001, independent noise seed 94001,
   5,000 present-target scenes/model. Save per-scene errors and exact radial
   and angular MSE contributions; inspect range-outlier and top-1% MSE shares.
5. Build evidence CSVs, a report addendum and a local reproducibility archive.

Final report integration, PDF compilation/visual inspection, editable presentation
creation/inspection and course-rubric verification follow the completed experiments.
They are not represented as already finished. The final rubric, deadline and page
limit have been requested. No publishing or team messages are authorized or queued.
