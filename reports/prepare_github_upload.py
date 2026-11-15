"""Prepare the authorized project snapshot for muneefo/DL_FINAL_REPORT."""
from pathlib import Path
import hashlib
import json
import re
import shutil

root = Path(__file__).resolve().parents[1]
target = root / 'github-final-report'
assert (target / '.git').is_dir(), 'Clone the destination repository first'
skip_names = {'.git', '__pycache__', 'node_modules', '.vscode'}
skip_files = {'reports/course_rubric_extracted.txt'}
paths = []
for directory in ['lib', 'methods', 'reports', 'results', 'deliverables', 'final-report-work']:
    for path in (root / directory).rglob('*'):
        if not path.is_file():
            continue
        if any(part in skip_names or part.startswith('.mplconfig') for part in path.parts):
            continue
        relative = path.relative_to(root).as_posix()
        if relative in skip_files or path.suffix in {'.pyc', '.tmp'}:
            continue
        paths.append(path)
for filename in ['LICENSE', 'Resume-Project.ps1', 'SSLISAC-main.zip', 'results1.zip']:
    if (root / filename).is_file():
        paths.append(root / filename)

secret_pattern = re.compile(
    r'gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{40,}|'
    r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----|AKIA[A-Z0-9]{16}'
)
text_extensions = {'.py', '.ps1', '.md', '.json', '.csv', '.txt', '.tex', '.log', '.yml', '.yaml', '.ini'}
suspects = []
for path in paths:
    if path.suffix in text_extensions and secret_pattern.search(path.read_text(encoding='utf-8', errors='replace')):
        suspects.append(str(path.relative_to(root)))
if suspects:
    raise RuntimeError('Potential credentials require review in these files: ' + ', '.join(suspects))

entries = []
for path in sorted(set(paths)):
    relative = path.relative_to(root)
    destination = target / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    entries.append(dict(path=relative.as_posix(), bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
shutil.copy2(root / 'README.md', target / 'README.original.md')
shutil.copy2(root / 'final-report-work/requirements.txt', target / 'requirements.txt')
(target / '.gitignore').write_text('''__pycache__/
*.py[cod]
.venv/
.plot-deps/
.pdf-deps/
.mplconfig*/
.vscode/
.env
.env.*
*.tmp
.DS_Store
''', encoding='utf-8')
(target / '.gitattributes').write_text('*.zip filter=lfs diff=lfs merge=lfs -text\n', encoding='utf-8')
(target / 'README.md').write_text('''# CISC 867 — ISAC final report project

Code, experiment outputs, model checkpoints and report sources for the
semi-supervised integrated sensing and communication project.

## Current status

The scheduled experiments finished on November 13, 2026. This snapshot includes
the 85,000-update reproductions, active-calibration controls, hardware/coverage
study, independent evaluation, supervised-fraction sweep, SNR transfer tests,
and independent localization outputs.

The final report PDF, presentation and demo still require completion and review.
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
''', encoding='utf-8')
(target / 'UPLOAD_MANIFEST.json').write_text(json.dumps(dict(
    destination='muneefo/DL_FINAL_REPORT', files=entries,
    file_count=len(entries), total_bytes=sum(e['bytes'] for e in entries),
    excluded='Git internals, installed dependencies, bytecode, rendering caches, editor settings, temporary files and extracted course specification'), indent=2), encoding='utf-8')
print(f'Prepared {len(entries)} project files, {sum(e["bytes"] for e in entries)/1e6:.1f} MB')
print('Credential-pattern scan passed')
