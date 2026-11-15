"""Build the updated self-contained LaTeX report from saved experiment evidence."""
from pathlib import Path
import re
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SOURCE = Path(r'C:\Users\SWIMS\.codex\attachments\048d05f3-7742-4a97-8bf4-252811fde9f4\Pasted text.txt')
RESULTS = ROOT / 'final-report-work/methods/results'
HARDWARE = RESULTS / 'hardware_study_20261108-160301'
text = SOURCE.read_text(encoding='utf-8-sig')


def replace_once(old, new):
    global text
    assert text.count(old) == 1, old[:100]
    text = text.replace(old, new, 1)


def replace_between(start, end, new):
    global text
    a, b = text.index(start), text.index(end, text.index(start))
    text = text[:a] + new + '\n\n' + text[b:]


abstract = r'''We study identifiability-guided position-label acquisition for semi-supervised integrated sensing and communication (ISAC). Extending our midterm reproduction, we complete supervised (SL), unsupervised (UL), and sequential semi-supervised (SSL) training at 85,000 updates. SSL uses 1,000 SL and 84,000 UL updates and reaches misdetection probability 0.05841 and conditional position RMSE 2.180 m, compared with 0.05781 and 2.294 m for full SL: similar detection with 98.8\% fewer supervised updates in this single-run comparison. We repair checkpoint aliasing and isolate acquisition randomness, then complete 50 matched active-calibration runs across five optimization seeds, followed by transfer and central-sector detection/SER evaluation. Guided acquisition has lower mean calibration error than random selection, but its paired interval includes zero; central-sector detection is nearly identical. A further 126 training runs span three independent array draws, two angular coverages, three optimization seeds, and seven variants. Weak-direction, full-trace, D-optimal, and noRC variants select identical labels in all 72 comparisons. Reconstructed candidate information gains are nearly proportional, explaining the lack of criterion separation. Fixed-label controls match or improve mean outcomes, and the guided-versus-random intervals include zero. Thus SSL reproduction is supported, while a distinct active-acquisition advantage is not established. Remaining work includes a labeled-fraction sweep, SNR variation, and separately calibrated operating-point evaluation.'''
replace_between(r'\begin{abstract}', r'\end{abstract}', '\\begin{abstract}\n' + abstract)
replace_once('Baseline Reproduction and Preliminary Experiments}', 'Baseline Reproduction and Extended Evaluation}')
replace_once(r'Midterm report}', r'Updated midterm report: experiments through November 8, 2026}')
replace_between(r'\begin{enumerate}', r'\end{enumerate}', r'''\begin{enumerate}
\item A completed 85,000-update SL/UL/SSL comparison in addition to the original 10,000-update reproduction.
\item Repaired acquisition experiments with fixed-label, full-trace and noRC controls, and detection/SER evaluation across five optimization seeds.
\item A 126-run robustness study over independent hardware draws and angular coverage, and diagnostics explaining identical information-based selections. These experiments do not establish a unique benefit from weak-direction targeting.
''')
replace_once('Disabling redundancy control (no redundancy control, noRC) is implemented but not yet run.',
             'Disabling redundancy control (noRC), a full-trace projection control, and a fixed-label training control are evaluated in the completed extension below.')
replace_once(r'\section{Results and Failure Analysis}', r'''\section{Original Midterm Results and Failure Analysis}
This section preserves the original midterm cohort and its numerical results. It is historical evidence, not the corrected acquisition cohort. The completed follow-up in Section~\ref{sec:extended} supersedes pending statements and tests the acquisition claims under repaired random streams and independent hardware draws.''')
replace_once('Shading marks training beyond our completed budget.', 'Shading marks training beyond the original midterm budget; the subsequent 85,000-update runs are reported separately.')
replace_once(r'\emph{Training budget.} Our runs stop at 10,000 updates,', r'\emph{Original training budget.} The midterm runs stopped at 10,000 updates,')
replace_once('Its 85,000-update label-efficiency result (1,000 SL plus 84,000 UL) remains unverified.',
             r'The subsequent 1,000-SL/84,000-UL run is reported in Section~\ref{sec:extended}; it provides a matched-total-update comparison, not a unique-label-efficiency curve.')
replace_between(r'\subsection{Runtime and limitations}', r'\section{Progress and Completion Plan}', r'''\subsection{Historical runtime and limitations}
The original pilot required 711--736 s of training per run on a laptop CPU. Its parameter histories stored NumPy views, so round snapshots aliased the final tensor. Acquisition also consumed the global training random stream. Scalar pilot metrics remain historical observations, but corrected matched comparisons are the basis for the updated interpretation below. The original seed-0 initialization discrepancy is not used as evidence of a method difference; the repaired smoke tests verify identical initial models and labels.

The original five seeds shared one array and fixed scene pools, and used 500 validation scenes. The follow-up adds independent test noise, three hardware draws, and two coverages, but still does not provide an independent localization test pool, separately calibrated detection thresholds, or a complete reference-label cost accounting. The original threshold routines calibrate on the evaluated observations and sort metric arrays independently before interpolation. We retain that protocol for comparability and describe its outputs as measured frontiers, not deployment guarantees.''')

# Read actual saved data: do not transcribe numerical tables by hand.
baselines = {}
for name, filename in [('SL', 'SL_85000.npz'), ('UL', 'UL_85000.npz'), ('SSL', 'SSL_SL_UL_1000.npz')]:
    with np.load(ROOT / 'results' / filename) as z:
        baselines[name] = {k: z[k].tolist() for k in z.files}
        assert len(z.get('loss_super', [])) + len(z.get('loss_unsuper', [])) == 85000
with (HARDWARE / 'summary.csv').open() as stream:
    hardware_rows = list(csv.DictReader(stream))
with (RESULTS / 'followup_summary/central_detection.csv').open() as stream:
    central_rows = list(csv.DictReader(stream))
with (RESULTS / 'summary_th10_cpu/summary.csv').open() as stream:
    pilot_rows = list(csv.DictReader(stream))
assert json.loads((HARDWARE / 'manifest.json').read_text())['finished']

section = r'''\section{Completed Follow-up Experiments}\label{sec:extended}
\subsection{Matched 85,000-update reproduction}
SL, UL and sequential SSL were each run for 85,000 total updates with 64 antennas, 256 subcarriers, the fixed impaired array, batch size 3,000, and the original full-sample test protocol (1.2 million samples per evaluation). SL uses the configured learning-rate decay at 50,000 updates; UL uses its configured constant learning rate. Table~\ref{tab:85k} reports the final sensing evaluation. These local runs use seed 42, whereas the original report's reproduction used seed 10, so differences from the historical 10,000-update cohort cannot be attributed to duration alone.

\begin{table}[t]\centering\small
\caption{Final sensing evaluation at 85,000 total updates. Single runs; lower is better for $P_{\rm md}$ and RMSE.}\label{tab:85k}
\begin{tabular}{lrrrr}\toprule
Method & SL updates & UL updates & $P_{\rm md}$ & RMSE (m)\\\midrule
'''
for name, z in baselines.items():
    section += f"{name} & {len(z.get('loss_super', [])):,} & {len(z.get('loss_unsuper', [])):,} & {1-z['pd_inter'][-1]:.5f} & {z['rmse_pos_inter'][-1]:.3f} " + r'\\' + '\n'
section += r'''\bottomrule\end{tabular}\end{table}

SSL's detection is close to SL's, and its position RMSE is slightly lower, with 98.8\% fewer supervised updates. This supports the reproduced label-sparing training behavior, but does not establish statistical equivalence or savings in unique acquired positions. UL remains near its earlier detection plateau. The supervised-update fraction is not the same experimental quantity as the number of queried scenes in the active study.

\begin{figure}[t]\centering
\begin{tikzpicture}\begin{axis}[reportaxis,xlabel={Training updates},ylabel={$P_{\rm md}$},ymode=log,
legend style={at={(0.98,0.98)},anchor=north east},xmin=0,xmax=85000]
'''
for name, color in [('SL', 'blue'), ('UL', 'orange!85!black'), ('SSL', 'red!75!black')]:
    z = baselines[name]
    section += '\\addplot[' + color + ',mark=*,mark size=0.8pt] coordinates {\n'
    section += '\n'.join(f'({x:g},{1-y:.9g})' for x, y in zip(z['num_iterations'], z['pd_inter']))
    section += '\n};\\addlegendentry{' + name + '}\n'
section += r'''\end{axis}\end{tikzpicture}
\caption{Completed 85,000-update sensing learning curves, embedded directly from the saved arrays. SSL switches to UL after 1,000 updates.}\label{fig:85k}
\end{figure}

\subsection{Repaired acquisition protocol and controls}
The corrected protocol copies each saved spacing vector and restores CPU, CUDA and NumPy random states around acquisition and validation. Rules share initial labels, model initialization, minibatch generators and update counts for a given seed. Smoke checks verify distinct round snapshots and matched initial states. A synthetic anisotropic example verifies that weak and full-trace criteria can select different candidates; they are not implementation aliases.

The existing pilot preset uses 2,000 training scenes and 500 validation scenes, 10 initial labels, five new labels per round up to 60, and 100 SL plus 100 UL updates per round (including initialization). The fixed-label control retains only the initial 10 labels while receiving all 11 training rounds. A full-trace control uses the same information matrices and gauge removal as weak projection but retains all directions. D-optimal remains a separate scalar criterion; noRC disables within-batch information updates. Fifty runs cover ten variants over optimization seeds 0--4, including the original six rules, full trace, fixed labels, and weak/full noRC. This preset runs on the available RTX 4090; its name \texttt{cpu} denotes the experiment size, not the execution device.

On the corrected fixed-array cohort, weak selection gives mean final calibration error 0.1527 versus 0.1714 for random and 0.1683 for fixed labels. The paired weak-minus-random difference is $-0.0187\pm0.0259$ (95\% halfwidth), so the interval includes zero. Acquisition time is about 24.0 s per weak run versus 26.2 s for D-optimal and approximately 0.003 s for random; these GPU-assisted measurements are not directly comparable to historical laptop CPU timings.

\subsection{Detection/SER evaluation and central-sector control}
Final checkpoints were evaluated with a shared, independent evaluation random seed (54321), 9,984 samples per beam and a $4\times4$ beam grid. The reported scalar is the minimum measured $P_{\rm md}$ among grid points with measured SER at most 0.01, using nominal $P_{\rm fa}=0.01$. Beam selection, threshold calibration and evaluation use the same test observations under the inherited routine. Consequently, this is a descriptive frontier, not a separately validated operating point. Independent test noise does not remove that limitation.

In the original sensing sector $[-40^\circ,-20^\circ]$ with communication sector $[30^\circ,50^\circ]$, weak, random and fixed-label mean misdetection are 8.213\%, 8.564\% and 8.431\%, respectively. This is transfer outside the narrow training support. A second evaluation changes only the sensing sector to $[-10^\circ,10^\circ]$, within training support. It tests one central sector rather than averaging over the full random-sector training distribution.

\begin{table}[t]\centering\small
\caption{Central-sector misdetection at measured SER $\le1\%$. Intervals are 95\% halfwidths across five optimization seeds; differences are paired with random. Units are percent or percentage points (pp).}\label{tab:central}
\begin{tabular}{lrr}\toprule
Rule & $P_{\rm md}$ (\%) & Difference (pp)\\\midrule
'''
for row in central_rows:
    if row['rule'] not in ['weak', 'random', 'fixed', 'maxangle', 'diversity', 'uncertainty']:
        continue
    name = 'Guided' if row['rule'] == 'weak' else row['rule'].capitalize()
    section += f"{name} & ${100*float(row['mean_pmd']):.3f}\\pm{100*float(row['ci95_halfwidth']):.3f}$ & ${100*float(row['paired_difference_vs_random']):.3f}\\pm{100*float(row['paired_ci95_halfwidth']):.3f}$ " + r'\\' + '\n'
section += r'''\bottomrule\end{tabular}\end{table}
Guided and random central-sector detection are nearly identical (Table~\ref{tab:central}); the paired interval includes zero. The fixed-label control is also comparable. Intervals quantify optimization-seed variability on one array and fixed scene pools, not independent hardware or Monte Carlo uncertainty.

\subsection{Why the information criteria coincide}
Weak, full-trace, D-optimal and both noRC variants have identical queried labels and saved models across the five corrected seeds. We reconstructed the candidate sets at rounds 1, 5 and 10 for all five seeds; every reconstructed weak query batch matched the saved batch. Within each inspected round, label-information gains were nearly scalar multiples of a shared matrix, with maximum relative reconstruction residual $3.511\times10^{-9}$. The weak subspace had dimension one and weak/trace score correlation exceeded 0.999999999997.

If $\Delta\F_i\simeq c_i\mathbf M$ within a round, the weak and trace scores are approximately constant multiples of $c_i$, while D-optimal gain is monotone in $c_i$ for positive-semidefinite gains. This explains the observed ranking agreement, rather than demonstrating a general equivalence of criteria. The synthetic check distinguishes the implementations, and the evidence does not support a special benefit from weak-direction targeting or greedy updates in this setting.

\subsection{Independent hardware and coverage robustness}
We completed 126 additional training runs: three independent hardware seeds (3101--3103), two sector-centre coverages ($\pm10^\circ$ and $\pm60^\circ$), three optimization seeds (0--2), and seven variants (weak, trace, D-optimal, random, fixed, weak noRC, full noRC). Each variant shares training budgets and scene pools within a coverage setting. Hardware uses the reference alternating-sort construction from Gaussian spacing perturbations of standard deviation $\lambda/25$, accepting only positive spacings with strictly ordered physical element positions. A private random generator leaves training streams unchanged. All three draws were accepted on their first attempt. The fixed-array midterm cohort is not pooled with these hardware replicates.

Each final checkpoint also receives the common-sector ISAC evaluation above. That sensing sector is outside narrow training support but within broad support. We average the three optimization seeds within each array and construct 95\% Student-$t$ intervals across the three hardware means; paired differences use the same arrays. Three independent arrays give imprecise intervals, and do not characterize a broad population of hardware faults.

\begin{table}[t]\centering\small
\caption{Robustness means over three array draws, each averaging three optimization seeds. The fixed control keeps 10 labels; active rules use 60.}\label{tab:hardware}
\begin{tabular}{llrr}\toprule
Coverage & Rule & Calib. error & $P_{\rm md}$ (\%)\\\midrule
'''
def hw(coverage, rule, metric, key='mean'):
    row = next(r for r in hardware_rows if int(r['coverage']) == coverage and r['rule'] == rule and r['metric'] == metric)
    return float(row[key])

for coverage in [10, 60]:
    for rule, label in [('weak', 'Guided'), ('random', 'Random'), ('fixed', 'Fixed')]:
        section += f"$\\pm{coverage}^\\circ$ & {label} & {hw(coverage, rule, 'calibration_error'):.4f} & {100*hw(coverage, rule, 'pmd_at_ser_1pct'):.3f} " + r'\\' + '\n'
section += r'''\bottomrule\end{tabular}\end{table}

For narrow coverage, guided-minus-random calibration error is $-0.0226\pm0.0512$ and misdetection differs by $-1.363\pm2.116$ pp. For broad coverage, the corresponding differences are $+0.0375\pm0.1914$ and $+1.100\pm5.220$ pp. All four intervals include zero. Fixed labels have the lowest mean calibration error and misdetection in both settings (Table~\ref{tab:hardware}), so these results do not establish that acquiring additional labels improves on continued training with the initial set. The narrow fixed-minus-random detection interval narrowly excludes zero ($-1.598\pm1.585$ pp), but it is exploratory among multiple comparisons over only three arrays, not evidence for an acquisition benefit.

Across both coverages, all arrays and optimization seeds, the four information-based comparators match weak selection in all 72 queried-label comparisons. Broader coverage and independent hardware therefore do not resolve the criterion degeneracy in these experiments.

\subsection{Updated interpretation and limitations}
The completed reproduction supports the practical value of sequential SSL with sparse supervised updates. The acquisition extension yields an informative negative result: improved calibration averages in one setting do not establish improved ISAC performance, weak-direction targeting is not distinguished from simpler information criteria, and fixed-label training is competitive. We do not claim equivalence, guaranteed label savings, or universal failure of active calibration.

Active experiments use reduced training/evaluation budgets and must not be numerically compared as if they were the 85,000-update reproduction. Scene pools remain fixed across hardware draws; independent localization test pools and reference-cost accounting are still missing. Detection summaries reuse observations for operating-point selection and threshold calibration, share one evaluation seed, and are subject to the inherited interpolation routine. Optimization- and hardware-seed intervals do not include all Monte Carlo or data-generation uncertainty. No SNR sweep or fixed-total-budget labeled-fraction sweep has yet been completed.
'''
replace_once(r'\section{Progress and Completion Plan}', section + '\n\n' + r'\section{Progress and Completion Plan}')
replace_between(r'\section{Progress and Completion Plan}', r'\section{Contribution Declaration and Conclusion}', r'''\section{Progress and Remaining Work}
\begin{table}[t]\centering\small
\caption{Updated progress against proposal commitments.}\label{tab:progress}
\begin{tabularx}{\columnwidth}{lYY}\toprule
Milestone & Completed evidence & Remaining requirement\\\midrule
Reproduction & SL/UL/SSL at 85k; learning curves & Repeated full-budget seeds; known-array control\\
Detection/SER & Transfer and central-sector tests & Separate threshold/beam validation; Monte Carlo uncertainty\\
Label efficiency & SSL with 1,000 SL updates & Fixed-total-budget fraction sweep; unique-label costs\\
Acquisition controls & Fixed labels, full trace, noRC & A setting that separates information criteria, or a qualified negative conclusion\\
Robustness & Three arrays; two coverages & SNR variation; independent localization pools\\\bottomrule
\end{tabularx}\end{table}

The original Weeks 6--10 plan has progressed through the matched-budget benchmark, acquisition controls, coverage/hardware tests and active detection/SER evaluation. Table~\ref{tab:plan} replaces the historical schedule with outstanding work; it does not assign unverified new individual contributions.
\begin{table}[t]\centering\small
\caption{Remaining work after the completed follow-up.}\label{tab:plan}
\begin{tabularx}{\columnwidth}{lY}\toprule
Priority & Deliverable\\\midrule
1 & Freeze evidence, verify threshold/interpolation behavior, and choose beams and thresholds on validation data before independent test evaluation.\\
2 & Sweep supervised fractions at fixed total updates; distinguish queried positions from repeated supervised updates and count tuning references.\\
3 & Add SNR variation and independent localization test pools, with per-scene tail-error analysis and more hardware draws if feasible.\\
4 & Finalize reproducibility checks, limitations, contribution declarations, manuscript and presentation.\\\bottomrule
\end{tabularx}\end{table}
No additional experiments are represented as complete. The 176 corrected active-training runs comprise the 50-run fixed-array cohort and 126-run hardware study; their 226 checkpoint evaluations comprise 50 transfer, 50 central-sector and 126 hardware-study ISAC evaluations. These counts exclude smoke tests and do not imply independent hardware replicates for every run.''')
replace_once('Table~\\ref{tab:plan} summarizes the planned responsibilities for the remaining project stages.',
             'These statements describe the original team contributions; the team should verify and update individual contributions for the final submission. Table~\\ref{tab:plan} lists remaining deliverables.')
replace_between('We reproduced the SL, UL, and untrained baselines of~\\cite{ssl} within 5\\%', r'\appendices', r'''The completed 85,000-update experiment shows that sequential SSL reaches detection close to full SL with 1,000 rather than 85,000 supervised updates, while UL remains at a higher misdetection plateau. This is a single-run reproduction result, not a statistical equivalence claim. In contrast, corrected controls, central-sector testing and independent-array experiments do not establish a distinct active-acquisition advantage. Information criteria select identical scenes in every tested hardware/coverage comparison, consistent with the nearly proportional candidate gains diagnosed in reconstructed rounds. Fixed-label training remains competitive. The final report should present these as separate findings: a successful SSL reproduction and a qualified negative result for the proposed acquisition mechanism under the tested model and budgets.''')
replace_between(r'\section{Code and Reproducibility}', r'\begin{thebibliography}', r'''\section{Code and Reproducibility}
The original midterm implementation is available in the linked team repository. Updated experiments were performed in a separate local working copy, \texttt{final-report-work/}, preserving the parent reproduction outputs. These local modifications and artifacts must be packaged or committed by the team before claiming that the online repository reproduces the updated tables.

\emph{Full-budget reproduction.} The parent \texttt{results/} folder contains \texttt{SL\_85000.npz}, \texttt{UL\_85000.npz} and \texttt{SSL\_SL\_UL\_1000.npz}; \texttt{methods/show\_results.py} generates the matched-budget plots. The newly executed SL/UL runs used Python 3.12 with PyTorch 2.5.1+cu121 and an RTX 4090. The parent configuration uses seed 42. The original report's seed-10 results are retained as historical comparisons, not pooled with the new seed-42 cohort.

\emph{Corrected acquisition.} \texttt{methods/run\_final\_stage.py} runs the 50-run comparison, noRC controls, summary and transfer ISAC evaluation. Its cohort begins at timestamp \texttt{20261108-142922}; summaries must filter older results. \texttt{methods/run\_followup.py} performs candidate-matrix diagnostics and central-sector evaluation. \texttt{check\_active\_protocol.py} verifies smoke snapshots, shared initialization and the synthetic projection test.

\emph{Hardware study.} \texttt{methods/run\_hardware\_study.py} runs the hardware/coverage factorial. \texttt{lib/hardware.py} uses private hardware seeds; \texttt{check\_hardware.py} verifies reproducibility, array ordering, independent draws and unchanged training randomness. \texttt{ISAC\_RUN\_MODE}, \texttt{ISAC\_COVERAGE}, \texttt{ISAC\_HARDWARE\_SEED} and \texttt{ISAC\_RESULTS\_DIR} select configuration without source edits. The completed cohort is \texttt{hardware\_study\_20261108-160301}. Its manifest records job arguments, exit codes, times and source SHA-256 hashes; an environment snapshot records the interpreter and installed package versions. Every active checkpoint saves true hardware, initial/final queried labels, settings, independent round snapshots and timings. \texttt{summarize\_hardware\_study.py} averages optimizer seeds within hardware before constructing intervals.

\emph{Evaluation and evidence.} \texttt{evaluate\_active\_isac.py} saves per-beam detection, false-alarm, SER and RMSE arrays, test sectors, sample count and seed. It supports transfer and central-sector scenarios. The report embeds new learning curves and tables directly from saved arrays and CSV summaries using \texttt{reports/update\_report.py}; it requires no external image files or bibliography database. Complete clean-environment end-to-end reproduction, independent localization data, unique-label cost accounting and separately calibrated operating points remain outstanding.''')

# Replace the main comparison figure with complete measured curves, while
# preserving the historical 10k tables and embedded paper-reference data.
full_tables = ''
for name, z in baselines.items():
    for prefix, values in [('FullDetection', [1-y for y in z['pd_inter']]),
                           ('FullLocalization', z['rmse_pos_inter'])]:
        full_tables += '\\pgfplotstableread{\nx y\n'
        full_tables += '\n'.join(f'{x:g} {y:.12g}' for x, y in zip(z['num_iterations'], values))
        full_tables += '\n}\\' + prefix + name + '\n'
text = text.replace(r'\begin{document}', full_tables + '\n' + r'\begin{document}', 1)
main_figure = r'''\begin{figure}[t]\centering
\begin{tikzpicture}
\begin{groupplot}[reportaxis,group style={group size=1 by 2,vertical sep=0.8cm},
 xmode=log,xmin=100,xmax=85000,xtick={100,1000,10000,85000},
 xticklabels={$10^2$,$10^3$,$10^4$,$8.5\times10^4$}]
\nextgroupplot[ymin=0.04,ymax=0.19,ylabel={$P_{\rm md}$},
 legend style={at={(0.98,0.98)},anchor=north east},legend columns=2]
\addplot[blue,dashed] table {\PaperSL};\addlegendentry{Paper SL}
\addplot[blue,mark=*,mark size=1pt] table {\FullDetectionSL};\addlegendentry{Our SL}
\addplot[orange!85!black,dashed] table {\PaperUL};\addlegendentry{Paper UL}
\addplot[orange!85!black,mark=*,mark size=1pt] table {\FullDetectionUL};\addlegendentry{Our UL}
\addplot[red!75!black,mark=triangle*,mark size=1pt] table {\FullDetectionSSL};\addlegendentry{Our SSL}
\addplot[green!50!black,dashed] table {\PaperSSL};\addlegendentry{Paper SSL}
\nextgroupplot[ymin=0,ymax=7,ylabel={Position RMSE (m)},xlabel={Training updates},
 legend style={at={(0.98,0.84)},anchor=north east}]
\addplot[blue,mark=*,mark size=1pt] table {\FullLocalizationSL};\addlegendentry{Our SL}
\addplot[orange!85!black,mark=*,mark size=1pt] table {\FullLocalizationUL};\addlegendentry{Our UL}
\addplot[red!75!black,mark=triangle*,mark size=1pt] table {\FullLocalizationSSL};\addlegendentry{Our SSL}
\addplot[black!65,dashed] table {\BaselineLocalization};\addlegendentry{Untrained}
\end{groupplot}
\end{tikzpicture}
\caption{Completed SL, UL, and sequential SSL runs through 85,000 updates. Top: measured misdetection and approximate published SL/UL/SSL curves. Bottom: measured conditional position RMSE and the original untrained reference. SSL uses 1,000 SL followed by 84,000 UL updates. Historical 10,000-update results remain in Table~\ref{tab:repro}; the measured curves here use the full-run archives.}
\label{fig:learning}
\end{figure}'''
figure_pattern = r'\\begin\{figure\}\[t\].*?\\end\{figure\}'
matches = list(re.finditer(figure_pattern, text, re.S))
target = next(m for m in matches if r'\label{fig:learning}' in m.group())
text = text[:target.start()] + main_figure + text[target.end():]
# The full comparison now occupies the original figure; remove its redundant
# single-panel follow-up version.
matches = list(re.finditer(figure_pattern, text, re.S))
duplicate = next(m for m in matches if r'\label{fig:85k}' in m.group())
text = text[:duplicate.start()] + text[duplicate.end():]
text = text.replace('whereas~\\cite{ssl} trains for 85,000, so checkpoint comparisons end at 10,000 updates (Fig.~\\ref{fig:learning}, unshaded region).',
                    'whereas~\\cite{ssl} trains for 85,000. Table~\\ref{tab:repro} preserves those historical checkpoint comparisons; Fig.~\\ref{fig:learning} now displays the completed 85,000-update runs.')
text = text.replace('The completed SSL learning results are included in Fig.~\\ref{fig:learning};',
                    'The completed 85,000-update SSL learning results are included in Fig.~\\ref{fig:learning};')
text = text.replace('Table~\\ref{tab:85k} reports the final sensing evaluation.',
                    'Table~\\ref{tab:85k} reports the final sensing evaluation, and Fig.~\\ref{fig:learning} shows both full learning metrics alongside the published detection references.')

# Static structural checks; compilation is a separate requirement.
assert text.count(r'\begin{document}') == text.count(r'\end{document}') == 1
environments = re.findall(r'\\(begin|end)\{([^}]+)\}', text)
stack = []
for action, env in environments:
    if action == 'begin':
        stack.append(env)
    else:
        assert stack and stack.pop() == env, env
assert not stack
labels = re.findall(r'\\label\{([^}]+)\}', text)
assert len(labels) == len(set(labels)), 'Duplicate labels'
refs = re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', text)
assert set(refs) <= set(labels), set(refs)-set(labels)
(OUT / 'updated_report.tex').write_text(text, encoding='utf-8')
(OUT / 'report_evidence.json').write_text(json.dumps(dict(
    baselines={name: dict(pmd=1-z['pd_inter'][-1], rmse=z['rmse_pos_inter'][-1]) for name,z in baselines.items()},
    hardware_summary=hardware_rows, central_summary=central_rows), indent=2))
(OUT / 'UPDATE_NOTES.md').write_text('''# Updated report

`updated_report.tex` preserves the original midterm theory, literature review,
inline bibliography and historical data, and adds completed follow-up evidence.
New tables and the 85k learning figure are generated from saved outputs.

Changes: revised abstract/contributions/conclusion; historical cohort labeled;
85k SL/UL/SSL comparison; corrected controls and runtime; central and transfer
detection; acquisition degeneracy diagnosis; independent-array/coverage study;
updated completion plan, limitations and reproducibility appendix.

The source remains self-contained for Overleaf/pdfLaTeX. Compile twice.
Static environment/label/reference checks passed. No local TeX compiler was
available at generation time, so PDF compilation and page-layout review remain
unverified. No final-report page limit or course rubric was provided.

Do not claim statistically proven acquisition superiority, unique-label savings,
or that local updates are already published to the linked GitHub repository.
Original contribution declarations were preserved and flagged for team review.
''', encoding='utf-8')
print(f'Created {OUT / "updated_report.tex"}; static structure/reference checks passed')
