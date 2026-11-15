# Hardware and coverage robustness study

126 matched training runs: three hardware draws, two coverages, three optimizer seeds, seven variants.
Random spacing uses the reference sorted Gaussian draw with std=lambda/25 and rejects unordered element positions.
Intervals use three hardware means (averaging optimizer seeds within each hardware). With only three independent arrays, uncertainty is substantial.
Common sensing sector [-40,-20] degrees and communication sector [30,50] degrees: outside narrow training support, within broad support.
Detection is the minimum measured PMD on the 4x4 beam grid at measured SER <=1%; 9,984 samples/beam, shared test seed 54321.
This measured frontier does not establish a separately validated operating-point guarantee or Monte Carlo confidence interval.

If no beam satisfies the SER constraint, PMD is NaN. No missing hardware draw is dropped or imputed; group means/intervals remain undefined, and summary.csv records finite hardware counts.

| Coverage | Metric | Rule | Mean | 95% halfwidth | Paired difference vs random | Paired halfwidth |
|---|---|---|---:|---:|---:|---:|
| 10 | calibration_error | weak | 0.180832 | 0.073631 | -0.022601 | 0.051171 |
| 10 | calibration_error | trace | 0.180832 | 0.073631 | -0.022601 | 0.051171 |
| 10 | calibration_error | full | 0.180832 | 0.073631 | -0.022601 | 0.051171 |
| 10 | calibration_error | random | 0.203433 | 0.060019 | 0.000000 | 0.000000 |
| 10 | calibration_error | fixed | 0.173295 | 0.088581 | -0.030138 | 0.031252 |
| 10 | calibration_error | weak_noRC | 0.180832 | 0.073631 | -0.022601 | 0.051171 |
| 10 | calibration_error | full_noRC | 0.180832 | 0.073631 | -0.022601 | 0.051171 |
| 10 | pmd_at_ser_1pct | weak | 0.086313 | 0.004773 | -0.013630 | 0.021157 |
| 10 | pmd_at_ser_1pct | trace | 0.086313 | 0.004773 | -0.013630 | 0.021157 |
| 10 | pmd_at_ser_1pct | full | 0.086313 | 0.004773 | -0.013630 | 0.021157 |
| 10 | pmd_at_ser_1pct | random | 0.099943 | 0.016479 | 0.000000 | 0.000000 |
| 10 | pmd_at_ser_1pct | fixed | 0.083960 | 0.009713 | -0.015983 | 0.015851 |
| 10 | pmd_at_ser_1pct | weak_noRC | 0.086313 | 0.004773 | -0.013630 | 0.021157 |
| 10 | pmd_at_ser_1pct | full_noRC | 0.086313 | 0.004773 | -0.013630 | 0.021157 |
| 60 | calibration_error | weak | 0.199181 | 0.230127 | 0.037473 | 0.191394 |
| 60 | calibration_error | trace | 0.199181 | 0.230127 | 0.037473 | 0.191394 |
| 60 | calibration_error | full | 0.199181 | 0.230127 | 0.037473 | 0.191394 |
| 60 | calibration_error | random | 0.161708 | 0.056870 | 0.000000 | 0.000000 |
| 60 | calibration_error | fixed | 0.135867 | 0.115015 | -0.025841 | 0.080504 |
| 60 | calibration_error | weak_noRC | 0.199181 | 0.230127 | 0.037473 | 0.191394 |
| 60 | calibration_error | full_noRC | 0.199181 | 0.230127 | 0.037473 | 0.191394 |
| 60 | pmd_at_ser_1pct | weak | 0.101340 | 0.062635 | 0.011000 | 0.052197 |
| 60 | pmd_at_ser_1pct | trace | 0.101340 | 0.062635 | 0.011000 | 0.052197 |
| 60 | pmd_at_ser_1pct | full | 0.101340 | 0.062635 | 0.011000 | 0.052197 |
| 60 | pmd_at_ser_1pct | random | 0.090340 | 0.010665 | 0.000000 | 0.000000 |
| 60 | pmd_at_ser_1pct | fixed | 0.085982 | 0.009011 | -0.004358 | 0.017729 |
| 60 | pmd_at_ser_1pct | weak_noRC | 0.101340 | 0.062635 | 0.011000 | 0.052197 |
| 60 | pmd_at_ser_1pct | full_noRC | 0.101340 | 0.062635 | 0.011000 | 0.052197 |

Identical-query comparisons versus weak: 72/72.
Inspect acquisition_agreement.json for the per-array results and matrix diagnostics.
No acquisition advantage is established when the paired interval includes zero. Do not attribute benefits to weak-direction targeting when queried labels coincide.