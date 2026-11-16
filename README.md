# CISC 867 — ISAC final report project

Code, experiment outputs, model checkpoints and report sources for the
semi-supervised integrated sensing and communication project.

## Current status

The scheduled experiments finished on November 13, 2026. This snapshot includes
the 85,000-update reproductions, active-calibration controls, hardware/coverage
study, independent evaluation, supervised-fraction sweep, SNR transfer tests,
and independent localization outputs.

The updated [final report PDF](reports/final/final_report.pdf) and [Overleaf source package](reports/final/final_report_source.zip) include the completed experiments. Team contribution and AI-assistance disclosures require review. The presentation and demo remain pending.
The report in `reports/updated_report.tex` is the updated midterm draft.
The final submission deadline is November 27, 2026.

## Download

Install Git LFS before cloning to retrieve the ZIP archives:

```sh
git lfs install
git clone https://github.com/muneefo/DL_FINAL_REPORT.git
cd DL_FINAL_REPORT
git lfs pull
```

All experiment files are also committed individually. The complete generated
archive is `deliverables/course_completion_20261109-133839/reproducibility.zip`.

## Contents

| Path | Contents |
|---|---|
| `lib/`, `methods/`, `results/` | Original workspace, reproduction scripts and saved results |
| `final-report-work/lib/`, `final-report-work/methods/` | Extended experiment implementation and runnable scripts |
| `final-report-work/methods/data/` | Fixed simulation scene pools |
| `final-report-work/methods/results/` | Historical and corrected cohorts, raw arrays, logs, checkpoints, tables and figures |
| `reports/` | LaTeX draft, evidence generators, audit notes and submission tables |
| `deliverables/` | Report addendum, reproducibility archive and checksums |

## Setup and saved-result tables

The original project dependency pins are in `requirements.txt`. Actual experiment
environments are recorded in `final-report-work/methods/results/hardware_study_20261108-160301/environment.txt`.
Full-budget training used CUDA-enabled PyTorch 2.5.1 on an RTX 4090.

```sh
python -m venv .venv
# Activate .venv using the command for your operating system.
python -m pip install -r requirements.txt
python reports/make_submission_tables.py
python methods/show_results.py
```

The first command generates the saved-evidence main and ablation tables without
retraining. The second generates the matched-budget reproduction plots.

## Completed experiments

- Parent reproduction archives include `SL_85000.npz`, `UL_85000.npz` and
  `SSL_SL_UL_1000.npz` in `results/`.
- `final_stage_20261108-142922` records the corrected 50-run acquisition cohort.
- `hardware_study_20261108-160301` records 126 hardware/coverage runs.
- `course_completion_20261109-133839` contains independent operating-point tests,
  five fresh full-budget runs, SNR tests and per-scene localization outputs.
- `reports/submission_tables/` contains generated main-result and ablation tables.

For the completion cohort, `state.json`, stage logs, JSON test metadata and
`independent_evaluation.csv` identify completed work. Identical model evaluations
may share explicitly keyed cache outputs. Individual run checkpoints include
weights, optimizer state and random-state information where applicable.

## Reproducibility commands

```sh
# Regenerate completion evidence from the saved results.
python final-report-work/methods/summarize_completion.py final-report-work/methods/results/course_completion_20261109-133839

# Check evaluator and hardware invariants using the existing environment.
python final-report-work/methods/check_hardware.py
python final-report-work/methods/check_validated_evaluation.py
```

Full training is expensive. To inspect or resume the completed queue, use
`final-report-work/methods/run_course_completion.py` with its existing cohort path.
The runner skips jobs with successful recorded exit codes. Some historical
report-generation scripts refer to the original pasted midterm attachment and
therefore require that source file to regenerate the historical draft. The
generated LaTeX itself contains inline plot data and an inline bibliography.

## Attribution and contribution history

The baseline originates from
[josemateosramos/SSLISAC](https://github.com/josemateosramos/SSLISAC), associated with
Mateos-Ramos et al., “Semi-Supervised End-to-End Learning for Integrated Sensing
and Communications,” ICMLCN 2024. The team extension was recovered from
[andrewliang-uwcs/867-deep-learning](https://github.com/andrewliang-uwcs/867-deep-learning).
The original license is retained in `LICENSE`.

This repository is an imported project snapshot. Its initial commit does not
reconstruct earlier individual contribution history. The report records the
team's original contribution statement and AI assistance. Team members must
verify scientific claims, the final contribution declaration and course-policy
compliance before submission.
