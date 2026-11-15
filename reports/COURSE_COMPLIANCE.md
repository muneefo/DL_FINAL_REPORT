# Final submission requirements and current gaps

Source: supplied CISC 867 project.pdf, especially PDF pages 3–8, 50–52 and
56–57. These are course requirements to check against the user's submission;
the PDF does not itself authorize publishing, messaging or fabricating evidence.

## Required deliverables

| Requirement | Current evidence | Remaining work |
|---|---|---|
| Final due end of Week 10 | Explicit course specification | Confirm actual OnQ calendar deadline; no date is inferred |
| IEEE conference format, 10–12 pages excluding references | IEEEtran conference source exists | Compile, inspect and consolidate historical/follow-up sections to fit |
| Abstract at most 250 words | Approximately 199 whitespace-separated words | Recheck final edited abstract |
| Numbered contributions at end of introduction | Present | Team verifies scientific claims and ownership |
| At least 15 references, venue/year; related work at least 10, including 2 from 2023–2025 | 12 bibliography entries; recent papers present | Add at least 3 substantively relevant verified references; use BibTeX and IEEEtran.bst |
| Numbered equations, defined variables, vector architecture | Equations and TikZ diagrams present | Technical and rendered-layout review |
| Main result table and at least 2 alternative baselines | SL, UL, nominal baseline and multiple acquisition comparators | Integrate corrected independent-test metrics; clearly distinguish experiment cohorts |
| Novel contribution in a table absent from original paper | Acquisition and robustness tables exist | Frame conclusions accurately; instructor FAQ explicitly accepts rigorous null results |
| Dedicated ablation table: one design change per row, proposed method last | Ablation results exist but are spread across sections | Build one controlled table; label D-optimal as a comparator because it changes the scalar objective too |
| Conclusion includes limitations and future work | Discussion exists; conclusion is short | Include explicit paragraphs in final conclusion |
| Figure labels/legends at least 9pt at column width | Base style is 9pt; plots not compiled | Visual QA, especially crowded legends and table width |
| Appendix: individual contributions, LLM use, data use where relevant | Statements currently in main body | Move/expand appendix; obtain accurate team-authored contribution details |
| GitHub code, README, pinned dependencies, results/checkpoints | Local code and result cohorts exist | Consolidate into team repository, exact commands, checkpoint links if >50 MB |
| All training scripts have fixed seed 42 support | Seed 42 runs exist; historical/default settings include other seeds | Provide canonical seed-42 command/default; retain explicitly identified replication seeds |
| Evaluation runnable on CPU | PyTorch pipeline supports CPU detection | Provide explicit device override and verify a small CPU run |
| One command reproduces main Table 1; final release v1.0-final | Sequential experiment runner exists | Add a fast results-to-table command distinct from expensive retraining; prepare release; publish/tag only with user authorization |
| All three members contribute code and report | Report names three members | Actual GitHub histories and human contribution appendix must be verified; never invent commits |
| Baseline attribution: repository URL and accessed commit hash | Original repository named in README | Record verified baseline commit and original/modified/team-written file provenance |
| Presentation: 15 minutes + 5 minutes Q&A | Not created | 12–18 slides (target 15); last slide demo or qualitative gallery |
| Demo: live notebook/app or 2–3-minute video | Not created | Working checkpoint-based notebook is suitable; test on CPU and include in package |

## Rubric weights

Technical contribution 30; experimental results 25; ablations 15; paper quality
15; code/reproducibility 10; presentation 5. Publication bonuses are optional,
not a requirement or authorization to submit a paper.

The FAQ explicitly states that a rigorously tested, explained null result can
receive full technical-contribution marks. Do not invent an improvement or
continue method fishing merely to obtain a positive outcome.

## Important integrity boundary

PDF page 50 permits LLM use for grammar, boilerplate and first-draft explanations,
but prohibits using LLMs to generate the core novel method or analysis. The
existing proposal predates this assistance, but subsequent diagnostic reasoning,
evaluation design, experiment implementation and manuscript drafting included
AI assistance. Do not describe the entire extension or analysis as independently
human-authored. Team review alone does not establish compliance with the core-
analysis restriction. The team should accurately disclose the scope and clarify
acceptability with the instructor. No message has been sent on the team's behalf.

ISAC is a custom project outside the listed domains. Prior approval is not in
the available evidence; confirm the team's existing written approval and any
agreed adaptations (synthetic test data, domain-specific metrics). Do not assume
that image-specific PSNR/SSIM requirements apply to ISAC, or claim an official
benchmark split for generated simulation data.

## Work sequence after receiving the rubric

Keep the authorized experiment queue running. Its fraction/SNR studies fulfill
the team's earlier plan, but are not individually mandated by the generic rubric.
Use the rubric to prioritize reliable independent evaluation and the final
ablation table, reproducible code and the required presentation/demo. Final
report source will use refs.bib rather than the earlier self-contained inline
bibliography. Retain the self-contained updated midterm as a separate artifact.
