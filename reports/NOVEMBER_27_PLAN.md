# Submission plan — November 27, 2026

The exact OnQ cutoff time is not yet verified. All dates use America/Toronto.

## Compute queue

Finish the already-running seed-10 UL run. Then execute seed-42 supervised
allocations 100, 500, 1,000 and 5,000, each at 85,000 total updates and the
unchanged batch size. The remaining nine allocations from the original queue
are deferred. Existing references retain their actual provenance, and are not
represented as newly matched-seed repeats.

The existing trainer was preserved while only the coordinator was replaced.
The new coordinator adopts that specific live training PID and waits for its
exit; it does not start a duplicate trainer. Resume checkpoints remain intact.
The schedule is recorded in completion_priority_plan.json and in the live state.

Target compute completion: November 19, subject to measured progress. SNR,
independent localization and packaging follow automatically. If the timing
slips, review scope before adding more compute rather than silently changing
training budgets or manufacturing repeated-seed evidence.

## Submission preparation

- Through November 19: assemble saved main/ablation evidence, bibliography,
  reproducibility commands and demo structure alongside the experiment queue.
- November 20–23: finish result integration, source attribution, package and
  checkpoint links; consolidate the paper into 10–12 body pages.
- November 24–26: compile and inspect every PDF page; validate 12–18 editable
  slides and a working demo; rehearse 15 minutes plus 5 minutes of questions.
- November 27: team submission after verifying the actual cutoff time.

One-command table generation:
`python reports/make_submission_tables.py`

The tables are mechanically formatted saved evidence, not new scientific
interpretation. The team owns and reviews all methodological and analytical
claims, including compliance with the rubric's restriction on AI-generated
core methodology/analysis. No instructor contact, GitHub push, release or
submission is performed automatically.
