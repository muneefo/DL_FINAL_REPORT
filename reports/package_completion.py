"""Package completed code/data and create a report-ready evidence addendum."""
from pathlib import Path
import json
import sys
import csv
import hashlib
import zipfile

parent = Path(__file__).resolve().parents[1]
root = parent/'final-report-work'
cohort = Path(sys.argv[1]).resolve()
out = parent/'deliverables'/cohort.name
out.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader((cohort/'independent_evaluation.csv').open()))
requirements_file = parent/'reports/course_requirements.json'
requirements = json.loads(requirements_file.read_text()) if requirements_file.exists() else None
destination_file = parent/'reports/github_destination.json'
destination = json.loads(destination_file.read_text()) if destination_file.exists() else None
summary = [r for r in rows if r['stage'] == 'stage2_sweep_evaluation']
latex = [r'\section{Independent Evaluation and Completion Experiments}',
    r'The legacy results above are retained for reproduction. The following evaluation calibrates a null threshold, selects a beam using validation observations, and freezes both before testing on separate observations. Each frame draws its own target range, correcting the legacy shared-range batches. No independently sorted interpolation is used.',
    r'Null calibration, validation and test use 10,000, 10,000 and 20,000 frames, respectively, with distinct seeds. The validation SER constraint is 1\%; a test violation is reported rather than used to reselect a beam. Wilson intervals quantify test-frame detection uncertainty for fixed models, separately from hardware/optimization variability.',
    r'\begin{table}[t]\centering\small\caption{Fresh fixed-budget sweep: 85,000 updates per run. Test operating points were selected on validation data.}',
    r'\begin{tabular}{lrrr}\toprule Run & $P_{\rm md}$ & $P_{\rm fa}$ & SER\\\midrule']
for r in summary:
    stem = Path(r['source']).stem.replace('fraction_', '').replace('_total85000', '')
    label = stem.replace('_', r'\_')
    def fmt(key):
        return f'{float(r[key]):.5f}' if r[key] else '---'
    latex.append(f"{label} & {fmt('pmd')} & {fmt('pfa')} & {fmt('ser')} " + r'\\')
latex += [r'\bottomrule\end{tabular}\end{table}',
    r'The sweep uses fresh online training scenes; supervised observation counts equal supervised updates times batch size, and are not unique fixed-pool acquisition labels. Two optimization seeds do not establish statistical equivalence.',
    r'SNR evaluation changes test sensing SNR to 5, 10, 20 and 25 dB without retraining. Independent localization uses new scene seed 93001 and noise seed 94001, saves per-scene errors, and decomposes squared position error exactly into radial and angular contributions.',
    r'Full per-checkpoint values, test uncertainty, validation feasibility and SER violations are provided in the accompanying CSV/JSON evidence. Remaining publication checks are PDF layout, presentation review, course rubric compliance and the human contribution declaration.']
(out/'completion_addendum.tex').write_text('\n\n'.join(latex), encoding='utf-8')
instructions = f'''# Reproducibility package

Completion cohort: {cohort.name}
Run scripts with the recorded Python environment; GPU training requires CUDA PyTorch.
Sources are in final-report-work/lib and final-report-work/methods.
Scene pools, parent reproduction results and completion evidence are included.
Logs and state.json distinguish completed jobs from unverified final submission steps.

Resume queue:
py -3.12 final-report-work/methods/run_course_completion.py "{cohort}"

The supplied course rubric has been mapped in reports/COURSE_COMPLIANCE.md.
The final paper must use IEEE format, 10–12 body pages excluding references,
at least 15 references, and BibTeX with IEEEtran.bst. The self-contained
updated_report.tex is an updated midterm artifact, not yet a compliant final.
Integrate and inspect completion_addendum.tex before final submission.
Prepare 12–18 slides (target 15), a demo and the accurate team/LLM-use appendix.
The calendar deadline and custom-project approval remain unverified.
No GitHub upload or publication has been performed.
'''
if destination:
    instructions += f'\nFinal GitHub destination: {destination["url"]}\n'
(out/'README.md').write_text(instructions, encoding='utf-8')
members = []
for directory in [root/'lib', root/'methods']:
    members += list(directory.glob('*.py'))
members += list((root/'methods/data').glob('*.pt'))
members += [p for p in cohort.rglob('*') if p.is_file() and not p.name.endswith('_resume.pt')]
members += list((parent/'reports').glob('*.py')) + list((parent/'reports').glob('*.tex'))
members += [parent/'reports/COURSE_COMPLIANCE.md', requirements_file] if requirements else []
members += [destination_file] if destination else []
members += list((parent/'results').glob('*.npz'))
members += [out/'README.md', out/'completion_addendum.tex']
checksums = {}
with zipfile.ZipFile(out/'reproducibility.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(set(members)):
        relative = str(path.relative_to(parent))
        archive.write(path, relative)
        checksums[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
(out/'SHA256.json').write_text(json.dumps(checksums, indent=2))
print(f'Created {out}; report/presentation/rubric review still required')
