"""Assemble the final IEEE paper from the existing manuscript and saved evidence."""
from pathlib import Path
import csv
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports'
RESULTS = ROOT/'final-report-work/methods/results'
COHORT = RESULTS/'course_completion_20261109-133839'
OUT = REPORTS/'final'
OUT.mkdir(exist_ok=True)
old = (REPORTS/'updated_report.tex').read_text(encoding='utf-8')
rows = list(csv.DictReader((COHORT/'independent_evaluation.csv').open()))


def section(start, end):
    return old[old.index(start):old.index(end, old.index(start))]


def tex_table(caption, label, columns, header, body, wide=False):
    env = 'table*' if wide else 'table'
    return ('\\begin{'+env+'}[t]\\centering\\small\n\\caption{'+caption+'}\\label{'+label+'}\n'
            '\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{'+columns+'}\\toprule\n'+header+r'\\\midrule'+'\n'
            +'\n'.join(' & '.join(map(str, r))+r'\\' for r in body)
            +'\n\\bottomrule\\end{tabular}\n\\end{'+env+'}\n')


def fmt(value, digits=3):
    return f'{float(value):.{digits}f}'


def source_rows(stage):
    return {Path(r['source']).name:r for r in rows if r['stage']==stage}


main = source_rows('stage1_validated')
sweep = source_rows('stage2_sweep_evaluation')
allocations = [0,100,500,1000,5000]
sweep_rows = [sweep[f'fraction_sl{n}_total85000_seed{10 if n==0 else 42}.npz'] for n in allocations]
localization = {}
for file in (COHORT/'stage4_localization').glob('*.npz'):
    with np.load(file) as data:
        item = json.loads(str(data['summary_json']))
        localization[Path(item['source']).name] = item

preamble = old[:old.index(r'\begin{document}')]
preamble = preamble[preamble.index(r'\documentclass'):]
preamble = '% Final IEEE report. Compile with LaTeX and BibTeX using refs.bib.\n' + preamble
preamble = preamble.replace(r'\documentclass[conference]{IEEEtran}',
    r'\documentclass[conference]{IEEEtran}'+'\n'+r'\usepackage[T1]{fontenc}')
preamble = re.sub(r'\\title\{.*?\}\n\\author',
    lambda m:r'\title{Identifiability-Guided Active Calibration for Semi-Supervised ISAC:\\Reproduction, Controlled Evaluation, and Label-Budget Experiments}'+'\n'+r'\author', preamble, count=1, flags=re.S)
preamble = preamble.replace('Updated midterm report: experiments through November 8, 2026',
                             'Final report draft, November 16, 2026')
preamble = preamble.replace('https://github.com/andrewliang-uwcs/867-deep-learning',
                             'https://github.com/muneefo/DL_FINAL_REPORT')
preamble = preamble.replace(r'\usepackage[hidelinks]{hyperref}',
                             r'\usepackage{cite}'+'\n'+r'\usepackage[hidelinks]{hyperref}')

abstract = r'''Position labels can constrain hardware parameters that remain ambiguous in unlabeled integrated sensing and communication (ISAC) measurements. We reproduce sequential semi-supervised learning and evaluate identifiability-guided acquisition of target-position labels. At 85,000 total updates, archived SSL with 1,000 supervised updates reaches misdetection probability 0.05841 and conditional position RMSE 2.180 m under the reproduction protocol, compared with 0.05781 and 2.294 m for full supervision. We then correct evaluation leakage and shared-range test batches by independently calibrating thresholds, selecting beams on validation observations, and testing frozen operating points. A fresh seed-42 sweep gives test misdetection probabilities 0.1402, 0.1405, 0.1051 and 0.0894 for 100, 500, 1,000 and 5,000 supervised updates, respectively, with total updates fixed at 85,000. The 5,000-update model has detected-target position RMSE 1.649 m. Across 176 active-calibration training runs, controlled ablations and three independent array draws, information-based variants select identical labels in all 72 hardware-study comparisons. Reconstructed label-information gains are nearly proportional. Guided-versus-random hardware intervals include zero, and fixed-label training is competitive. Independent localization confirms that a small range-error tail dominates unconditional RMSE. The completed experiments support sparse-supervision training and document the acquisition mechanism's lack of demonstrated benefit under the tested model.'''

intro = section(r'\section{Introduction}', r'\section{Related Work}')
intro = intro[:intro.index('This midterm report makes three contributions:')]+r'''The completed project makes four contributions.
\begin{enumerate}
\item We reproduce SL, UL and sequential SSL at 85,000 updates and preserve the published-protocol comparison alongside the completed supervised-budget sweep.
\item We implement independent null calibration, validation-based beam selection and held-out operating-point tests, and document two defects in the inherited evaluator.
\item We evaluate fixed-label, full-trace and noRC controls across optimization seeds, independent hardware draws and angular coverage, with paired uncertainty and acquisition diagnostics.
\item We report sensing-SNR transfer and independent per-scene localization errors, providing a reproducible account of successful reproduction and the acquisition extension's null results.
\end{enumerate}
'''
related = section(r'\section{Related Work}', r'\section{System and Implemented Learning Method}')
related += r'''\emph{Optimization and uncertainty.} Our differentiable implementation uses PyTorch~\cite{pytorch} and Adam~\cite{adam}, following the original code's learning-rate settings. These software and optimizer choices support the model-based pipeline rather than defining the acquisition criterion. For independent test frames we report Wilson intervals~\cite{wilson} for detection and false alarms. These conditional test-frame intervals answer a different question from the Student-$t$ intervals over optimization seeds or independent hardware draws reported for the acquisition experiments. Both are needed because additional test frames do not create additional trained models or hardware replicates.
'''
system = section(r'\section{System and Implemented Learning Method}', r'\subsection{Simulation protocols and metrics}')
system = system.replace('Both use the Adam optimizer.', r'Both use the Adam optimizer~\cite{adam} in PyTorch~\cite{pytorch}.')
system += r'''\subsection{Metrics and physical settings}
We use $K=64$, $S=256$, carrier frequency 60 GHz, subcarrier spacing 120 kHz, a 720-point angle grid, and a 200-point range grid over $[0,200]$ m. Communication uses QPSK, five normalized exponentially weighted taps and SNR 20 dB. Default sensing SNR is 15 dB. Training sector centres are uniform over $[-60,60]^\circ$ for reproduction and the online fraction sweep. Active experiments also use the narrow $[-10,10]^\circ$ centre support, with sector widths uniform over $[10,20]^\circ$.

Detection metrics are $P_{\rm md}=P(\widehat t=0\mid t=1)$ and $P_{\rm fa}=P(\widehat t=1\mid t=0)$, with nominal false-alarm target 0.01. Communication SER is the fraction of incorrectly decoded symbols. The detected-target position metric is
\begin{equation}
 \mathrm{RMSE}_{\mathrm{det}}=\sqrt{\E[\|\widehat{\mathbf p}-\mathbf p\|_2^2\mid t=1,\widehat t=1]}.
\end{equation}
Localization-only tests evaluate all present scenes and report unconditional position RMSE and median position error. Their conditioning and angular distribution differ from the joint detection test, so both protocols are specified with every table.
'''
acquisition = section(r'\section{Identifiability-Guided Acquisition}', r'\section{Original Midterm Results and Failure Analysis}')
acquisition = acquisition.replace('Disabling redundancy control (noRC), a full-trace projection control, and a fixed-label training control are evaluated in the completed extension below.',
    'We also evaluate noRC, a full-trace control retaining every gauge-free direction, and a fixed-label control retaining only the initial 10 labels through the same training schedule.')
acquisition = acquisition.replace('Scoring consumes additional observations and draws from the global random generator, so equal seeds do not guarantee identical training-noise streams across rules.',
    'The corrected implementation restores CPU, CUDA and NumPy random states around scoring and validation. Scoring uses separately seeded measurements, and parameter histories store independent copies of every round. Mini-batch selection uses a private generator. These repairs were checked by replaying initial models and by an interrupted/resumed training smoke test.')
acquisition = acquisition.replace('We implement six acquisition rules:', 'The original comparator set contains six acquisition rules:')

protocol = r'''\section{Experimental Protocols}\label{sec:protocols}
\subsection{Experiment cohorts and budgets}
Table~\ref{tab:settings-final} records the budgets used in each cohort. The archived reproduction uses the original fixed array and online random training examples. SL minimizes position MSE for all 85,000 updates. UL maximizes the receiver peak objective for all updates. Archived SSL uses 1,000 SL and 84,000 UL updates. The new online sweep fixes the total at 85,000 and runs seed 42 with 100, 500, 1,000 and 5,000 SL updates, followed by UL. A fresh seed-10 UL reference was completed first. Existing full-SL and UL archives are additional reference checkpoints.

The primary supervised-budget comparisons use seed 42. The extra seed-10 UL run is explicitly identified because its initialization and training stream differ. The archived SSL file does not save its training seed, and its early learning curve matches the original midterm trace. We therefore report its stored outputs without assigning the archive a newly verified seed. Every fresh sweep checkpoint records seed, update allocation, batch size, rates, hardware and elapsed time.
'''
protocol += tex_table('Completed experiment configurations. The active preset executes on CUDA when available.', 'tab:settings-final', 'lrrr',
    'Setting & Reproduction & Active & Fresh sweep', [
    ['Total updates','85,000','2,200','85,000'], ['Batch size','3,000','64','3,000'],
    ['SL/UL rate',r'$(4/5)10^{-7}$',r'$(4/5)10^{-6}$',r'$(4/5)10^{-7}$'],
    ['Train scenes','Online','2,000 fixed','Online'], ['Initial/final labels','Online','10/60','Online'],
    ['Test frames/beam','1,200,000','9,984 legacy','20,000 final'],
    ['Beam grid',r'$8\times8$',r'$4\times4$',r'$4\times4$']])
protocol += r'''\subsection{Acquisition repetitions and controls}
The corrected fixed-array cohort runs ten variants across optimization seeds 0--4, totaling 50 runs. It includes weak, D-optimal, random, max-angle, diversity, uncertainty, full trace, fixed labels and weak/full noRC. All acquired labels are unique scene indices. The fixed-label control continues training on its original 10 labeled scenes through all 11 rounds, receiving the same 1,100 SL and 1,100 UL updates as an active rule.

The hardware study adds 126 runs from three independent array draws, two centre coverages, three optimization seeds and seven variants. The variants are weak, trace, D-optimal, random, fixed, weak noRC and full noRC. Hardware seeds 3101--3103 drive a private Gaussian draw with standard deviation $\lambda/25$, followed by the reference alternating-sort construction. Draws must have positive spacing parameters and strictly ordered induced element positions. All three proposed draws pass on their first attempt. Scene pools are shared across arrays within each coverage, keeping scene variation controlled.

Fixed-array intervals use five optimization seeds. Hardware-study intervals first average the three optimization seeds within each array, then use the three array means as independent replicates. Paired comparisons share the same seed or array. We use two-sided 95\% Student-$t$ intervals and retain their limited precision when only three arrays are available.

\subsection{Independent calibration, validation and test}
The inherited ISAC routine chooses thresholds on evaluated observations. It interpolates independently sorted PFA and PD/RMSE arrays, which discards the correspondence of individual operating points. In addition, the channel expands scalar angular limits to a batch but leaves scalar range limits unchanged. Consequently, its usual scalar-range call draws one range shared across a batch. We preserve those outputs for reproduction and implement a corrected evaluator for the completed tests below.

The corrected evaluator draws 10,000 absent-target calibration frames with seed 87001. With false-alarm target $\alpha_0=0.01$, the threshold is the order statistic at rank
\begin{equation}
 k=\lceil(n_0+1)(1-\alpha_0)\rceil,
\end{equation}
and a detection requires the metric to be strictly larger. If the rank exceeds the calibration size, the threshold is infinite. Under the additive-noise null, transmit-beam allocation does not affect the sensing observation, so calibration is shared across beams for a fixed receiver and sector. A replay test verifies that null-beam invariance.

Next, 10,000 validation frames at seed 88001 evaluate a $4\times4$ beam grid. Among beams with measured validation SER at most 1\%, we select the beam with lowest validation misdetection, breaking ties by SER and then grid index. If no beam is feasible, the output records this status. The selected beam and threshold are frozen before 20,000 test frames at seed 89001. Each frame receives an independently drawn range. We report measured test SER even when it exceeds the validation constraint, without changing the selected beam.

Wilson intervals quantify test-frame PMD and PFA uncertainty for each fixed operating point~\cite{wilson}. Symbol errors share a channel within each frame, so the saved SER standard error uses frame averages. Neither interval replaces repetition over trained models, hardware or scene pools. The original frontier protocol and this frozen-point test answer different evaluation questions, and the report labels their outputs accordingly.

\subsection{Generalization and independent localization}
Joint tests use sensing angles $[-40,-20]^\circ$ and communication angles $[30,50]^\circ$. For narrow active training, the sensing test is outside the training support. A separate central-sector evaluation uses $[-10,10]^\circ$ for sensing with the original communication sector. This central sector lies within narrow training support, although it is one fixed sector rather than the entire random-sector training distribution.

SNR transfer freezes model weights and changes test sensing SNR to 5, 10, 20 and 25 dB. The 15 dB sweep result completes the five-level curve. Threshold calibration and beam validation are rerun at each SNR, followed by independent testing. Communication SNR remains 20 dB. Thus the curves evaluate adaptation of the operating point at a changed sensing SNR, without retraining hardware parameters.

The localization-only evaluation draws 5,000 new present-target scenes with seed 93001 and fresh measurement noise with seed 94001. Sector centres follow each model's training coverage. It saves true and estimated angles/ranges, gain magnitudes and per-scene errors. This extends the original 500-scene validation metric with a separate scene pool and supports direct tail-error accounting.
'''

results = r'''\section{Results}\label{sec:results}
\subsection{Completed reproduction at 85,000 updates}
The original midterm checkpoint comparison reproduced SL and UL misdetection at 10,000 updates within 2.5\% and 0.5\% of approximate published-curve values. The completed full-budget archives appear in Table~\ref{tab:legacy-final}. Final sensing-only PMD is 0.05781 for SL, 0.09737 for UL and 0.05841 for archived SSL. The corresponding detected-target RMSEs are 2.294, 3.305 and 2.180 m. Figure~\ref{fig:learning} extends all measured learning curves to 85,000 updates.
'''
results += tex_table('Archived full-budget sensing evaluation using the inherited reproduction protocol. Each method is one trained archive.',
    'tab:legacy-final','lrrrr','Model & SL & UL & $P_{\rm md}$ & RMSE (m)',[
    ['SL','85,000','0','0.05781','2.294'],['UL','0','85,000','0.09737','3.305'],
    ['Archived SSL','1,000','84,000','0.05841','2.180']])
figure_match = next(m for m in re.finditer(r'\\begin\{figure\}\[t\].*?\\end\{figure\}', old, re.S)
                    if r'\label{fig:learning}' in m.group())
results += figure_match.group().replace('Historical 10,000-update results remain in Table~\\ref{tab:repro}; the measured curves here use the full-run archives.',
                                       'The measured curves use the full-run archives and the inherited sensing evaluation.').replace(
                                           r'\begin{groupplot}[reportaxis,', r'\begin{groupplot}[reportaxis,width=0.88\columnwidth,')+'\n\n'
results += r'''Archived SSL uses 98.8\% fewer supervised updates than full SL while reaching close detection and a slightly lower conditional RMSE in these stored runs. This is a supervised-update comparison. It does not measure the cost of repeatedly acquiring positions in the fixed-pool active experiment. The archive is one model per method, so the numerical closeness is reported descriptively.

\subsection{Main independent operating-point comparison}
Table~\ref{tab:main-final} evaluates the existing full-budget archives and two array controls with the corrected frozen-point test. The nominal control uses $\lambda/2$ in beam and receiver design. The known-array control uses the true fixed array without training. All models are tested at sensing SNR 15 dB, communication SNR 20 dB, and the same prescribed sectors. The selected beam can differ across models because selection uses each model's validation data.
'''
main_body=[]
for label, filename in [('Nominal array','nominal_reference.npz'),('Known array','known_reference.npz'),
    ('SL archive','SL_85000.npz'),('UL archive','UL_85000.npz'),('SSL archive','SSL_SL_UL_1000.npz')]:
    r=main[filename]; lo,hi=json.loads(r['pmd_ci95'])
    main_body.append([label,fmt(100*float(r['pmd']),2),f'[{100*lo:.2f},{100*hi:.2f}]',
                      fmt(100*float(r['pfa']),2),fmt(100*float(r['ser']),3),fmt(r['rmse_detected'],3)])
results += tex_table('Main independent-test results. PMD, PFA and SER are percentages. PMD intervals are Wilson intervals over test frames. RMSE is conditioned on detection.',
    'tab:main-final','lrrrrr',r'Model & PMD & 95\% CI & PFA & SER & RMSE (m)',main_body,wide=True)
results += r'''The archived SL and SSL checkpoints yield PMD 8.65\% and 8.73\%, respectively. The known-array control reaches 8.01\%, whereas nominal spacing gives 37.83\%. Measured PFA varies around the calibration target because test observations are independent. Differences from Table~\ref{tab:legacy-final} reflect changes in threshold/beam selection, independent range draws, sample size and test random streams. They should not be attributed to further training.

\subsection{Fresh supervised-budget sweep}
All fresh runs complete 85,000 updates. Table~\ref{tab:fractions-final} gives the four seed-42 SSL allocations and the extra seed-10 UL reference. The supervised minibatch contains 3,000 fresh scenes, so 100 supervised updates correspond to 300,000 labeled training observations. These observation counts differ from the 10--60 unique position labels repeatedly reused in active calibration.
'''
fraction_body=[]
for n,r in zip(allocations,sweep_rows):
    lo,hi=json.loads(r['pmd_ci95'])
    fraction_body.append([f'{n:,}',10 if n==0 else 42,fmt(100*float(r['pmd']),2),
        f'[{100*lo:.2f},{100*hi:.2f}]',fmt(100*float(r['pfa']),2),fmt(r['rmse_detected'],3)])
results += tex_table('Fresh 85,000-update sweep with independent operating-point evaluation. Probabilities are percentages and PMD intervals are test-frame Wilson intervals.',
    'tab:fractions-final','rrrrrr',r'SL updates & Seed & PMD & 95\% CI & PFA & RMSE (m)',fraction_body,wide=True)
results += r'''At seed 42, PMD is 14.02\% with 100 supervised updates and 14.05\% with 500. It falls to 10.51\% with 1,000 and 8.94\% with 5,000. Conditional position RMSE is 4.178, 4.562, 3.225 and 1.649 m, respectively. Within this seed's completed allocations, the larger supervised budgets produce lower PMD after the 500-update point. The small difference between 100 and 500 updates does not establish a strict ordering. The 1,000-update fresh model also differs from the archived 1,000-update SSL checkpoint, motivating transparent checkpoint/seed provenance and repetition before population-level label-efficiency claims.
'''


def panel_plot(plot_type):
    lines=[r'\begin{figure}[t]\centering\begin{tikzpicture}',
        r'\begin{groupplot}[reportaxis,group style={group size=1 by 2,vertical sep=0.8cm}]']
    colors=['blue','orange!80!black','green!45!black','red!70!black','violet']
    if plot_type=='fraction':
        for key,ylabel in [('pmd',r'$P_{\rm md}$ (\%)'),('rmse_detected','Position RMSE (m)')]:
            xlabel = ',xlabel={Supervised updates}' if key=='rmse_detected' else ''
            lines.append(r'\nextgroupplot[xmin=-0.3,xmax=4.3,xtick={0,1,2,3,4},xticklabels={0,100,500,1000,5000},ylabel={'+ylabel+'}'+xlabel+']')
            scale=100 if key=='pmd' else 1
            lines.append(r'\addplot[blue,mark=*] coordinates {'+' '.join(f'({i},{float(r[key])*scale:.9g})' for i,r in enumerate(sweep_rows) if i>0)+'};')
            lines.append(r'\addplot[orange!80!black,only marks,mark=square*] coordinates {(0,'+f'{float(sweep_rows[0][key])*scale:.9g}' +')};')
            if key=='pmd':
                lines.extend([r'\addlegendentry{SSL, seed 42}',r'\addlegendentry{UL, seed 10}'])
        caption='Fresh supervised-budget sweep at 85,000 total updates. The horizontal positions denote discrete tested allocations. Joint test thresholds and beams are selected before testing. The seed-10 UL point is an additional reference.'
        label='fig:fraction-final'
    else:
        for key,ylabel in [('pmd',r'$P_{\rm md}$ (\%)'),('rmse_detected','Position RMSE (m)')]:
            xlabel = ',xlabel={Test sensing SNR (dB)}' if key=='rmse_detected' else ''
            lines.append(r'\nextgroupplot[xmin=5,xmax=25,xtick={5,10,15,20,25},ylabel={'+ylabel+'},legend columns=3,legend style={at={(0.5,1.02)},anchor=south}'+xlabel+']')
            for i,n in enumerate(allocations):
                name=f'fraction_sl{n}_total85000_seed{10 if n==0 else 42}.npz'
                series=[r for r in rows if Path(r['source']).name==name and r['stage'] in ['stage2_sweep_evaluation','stage3_snr']]
                series.sort(key=lambda r:float(r['sensing_snr_db']))
                scale=100 if key=='pmd' else 1
                lines.append('\\addplot['+colors[i]+',mark=*] coordinates {'+' '.join(f"({float(r['sensing_snr_db']):g},{float(r[key])*scale:.9g})" for r in series)+'};')
                if key=='pmd':
                    lines.append(r'\addlegendentry{'+ ('UL' if n==0 else f'SL {n:,}')+'}')
        caption='Sensing-SNR transfer of the completed sweep checkpoints. The legend gives supervised updates for SSL. UL uses seed 10 and SSL uses seed 42. Communication SNR remains 20 dB. A separate validation-selected operating point is frozen before each SNR test.'
        label='fig:snr-final'
    lines += [r'\end{groupplot}\end{tikzpicture}',r'\caption{'+caption+r'}\label{'+label+'}',r'\end{figure}']
    return '\n'.join(lines)


results += panel_plot('fraction')
results += r'''\subsection{Sensing-SNR transfer}
Figure~\ref{fig:snr-final} evaluates frozen sweep models at five sensing SNRs. At low SNR, detection and localization deteriorate for every allocation. At 15 dB, the 5,000-supervised-update model attains the lowest PMD of the tested fresh models. The saved SNR arrays also report PFA, SER and detected-target RMSE at each frozen operating point. PFA can remain identical across sensing SNR for a given model because the absent-target null contains noise and the same calibration/test random seeds are reused. Communication SER changes little when only sensing SNR changes, although beam validation can choose another operating point.

The transfer test does not retrain the model at each noise level. It therefore measures the deployment of a fixed learned array with validation-based operating-point adjustment. Common random streams improve comparability between checkpoints but do not create multiple independent Monte Carlo replicates.
'''
results += panel_plot('snr')
results += r'''\subsection{Independent localization and error tails}
Table~\ref{tab:localization-final} reports the new 5,000-scene localization pool. It evaluates all present scenes over the broad training-sector distribution. The exact squared-position decomposition is
\begin{equation}
 \|\widehat{\mathbf p}-\mathbf p\|_2^2
 = (\widehat R-R)^2+2\widehat R R[1-\cos(\widehat\theta-\theta)].
 \label{eq:decomp}
\end{equation}
This permits direct accounting rather than an angular-error bound. A range outlier is defined by $|\widehat R-R|>2.5\Delta R$, with $\Delta R=c/(2S\Delta_f)\simeq4.88$ m. The top-1\% statistic is the share of total position MSE contributed by the 50 largest errors in the 5,000-scene pool.
'''
loc_body=[]
for label,name in [('UL, seed 10','fraction_sl0_total85000_seed10.npz'),
    ('SSL 100','fraction_sl100_total85000_seed42.npz'),('SSL 500','fraction_sl500_total85000_seed42.npz'),
    ('SSL 1,000','fraction_sl1000_total85000_seed42.npz'),('SSL 5,000','fraction_sl5000_total85000_seed42.npz'),
    ('Known array','known_reference.npz')]:
    r=localization[name]
    loc_body.append([label,fmt(r['rmse']),fmt(r['median']),fmt(100*r['radial_mse_share'],1),
        fmt(100*r['tail_fraction'],2),fmt(100*r['top_one_percent_mse_share'],1)])
results += tex_table(r'Independent present-scene localization. Radial is the radial share of position MSE. Tail is the fraction of scenes exceeding the range-outlier threshold. Top 1\% is their largest-error MSE share.',
    'tab:localization-final','lrrrrr',r'Model & RMSE (m) & Median (m) & Radial (\%) & Tail (\%) & Top 1\% (\%)',loc_body,wide=True)
results += r'''For the fresh SSL checkpoints, unconditional RMSE remains 13.53--15.48 m while median errors range from 0.402 to 0.768 m. Radial error contributes 97.0--98.5\% of position MSE. Only 2.34--3.34\% of scenes exceed the range-outlier threshold, yet range-tail scenes contribute over 99\% of position MSE for those models. The top 1\% of errors contributes 73.2--82.4\% of total MSE. Even known hardware retains RMSE 13.482 m with median 0.334 m, showing that accurate array parameters alone do not remove the localization tail in this receiver.

The saved gain magnitudes support the historical weak-echo diagnosis. For SSL with 5,000 SL updates, the median gain is 0.074 among range-outlier scenes and 0.599 among other scenes. This is an observed association in one independent scene pool. It does not isolate the causal effect of gain or show that calibration cannot alter peak selection, because angular and range estimation share the joint map.
'''

ablation = r'''\section{Acquisition Ablations and Robustness}\label{sec:ablations}
\subsection{Controlled fixed-array ablations}
The corrected 50-run cohort gives final calibration error $0.1527\pm0.0237$ for weak acquisition, $0.1714\pm0.0117$ for random, and $0.1683\pm0.0241$ for fixed labels, where values are mean and 95\% halfwidth across five optimization seeds. The paired weak-minus-random calibration difference is $-0.0187\pm0.0259$, so its interval includes zero.

Table~\ref{tab:ablation-final} isolates the weak-subspace restriction, within-batch greedy update, and acquisition of new labels. Each row changes one component relative to the proposed method in the final row. D-optimal remains a comparator rather than the projection ablation because it changes the scalar objective as well. The calibration and central-sector legacy-frontier summaries correspond to the completed corrected cohort, preserving the evaluation protocol used when those controls were run.
'''
cal={r['rule']:r for r in csv.DictReader((REPORTS/'submission_tables/ablation_results.csv').open())}
abl_body=[]
for key,label in [('trace','Full trace'),('weak_noRC','Greedy updates off'),('fixed','New labels off'),('weak','Proposed method')]:
    r=cal[key]
    abl_body.append([label,r['labels'],f"${float(r['calibration_mean']):.4f}\\pm{float(r['calibration_ci_halfwidth']):.4f}$",
                     f"${float(r['pmd_percent_mean']):.3f}\\pm{float(r['pmd_percent_ci_halfwidth']):.3f}$"])
ablation += tex_table(r'One-change acquisition ablations. All rows receive 2,200 updates. PMD uses the central-sector measured-frontier protocol. Entries are means with 95\% halfwidths across five seeds.',
    'tab:ablation-final','lrrr',r'Variant & Labels & $e_d$ & PMD (\%)',abl_body)
ablation += r'''Central-sector PMD is 8.729\% for weak, 8.736\% for random and 8.740\% for fixed labels. The paired weak-minus-random difference is $-0.008\pm0.111$ percentage points. Thus the central test does not establish a detection advantage for acquisition. Uncertainty selection has mean calibration error 0.3017 and central PMD 10.453\%, with a wide PMD interval across seeds. Its historical selection of low-gain, near-broadside scenes helps explain why prediction uncertainty need not prioritize calibration information.

The independent evaluator was also run on every final active checkpoint, including nominal/known controls. Full per-checkpoint outputs are published in the completion CSV and JSON records. Their corrected operating-point values should be used for any later claim about deployment. Table~\ref{tab:ablation-final} is retained as the direct component-removal comparison from the cohort's original evaluation.

\subsection{Information-matrix diagnostic}
Weak, full-trace, D-optimal and both noRC variants have identical queried indices and saved round models across the corrected five seeds. We reconstruct candidate sets at rounds 1, 5 and 10 for all seeds. Every reconstructed weak batch matches the saved queries. Candidate label gains in each inspected round are nearly proportional to a shared matrix, with maximum relative residual $3.511\times10^{-9}$. The weak subspace has dimension one and weak/trace score correlation exceeds 0.999999999997.

For an approximate form $\Delta\F_i=c_i\mathbf M$, weak and full-trace scores are constant multiples of $c_i$. D-optimal gain is monotone in the same coefficient for positive-semidefinite increments. This explains the observed ranking agreement in the inspected rounds. A synthetic anisotropic matrix test makes weak and trace select different candidates, verifying that the implementations can distinguish directions. The experiment therefore provides no empirical separation of the weak-subspace and greedy components under these simulated conditions.

\subsection{Hardware and coverage results}
Table~\ref{tab:hardware-final} reproduces the completed hardware-study means. Its detection values use the common sensing sector $[-40,-20]^\circ$ under the legacy measured-frontier evaluation. Three optimization seeds are averaged within each of three independent arrays. The broad-coverage study contains the test sector within its support, whereas narrow training is evaluated in transfer.
'''
hardware=list(csv.DictReader((RESULTS/'hardware_study_20261108-160301/summary.csv').open()))
hw_body=[]
for coverage in [10,60]:
    for rule,label in [('weak','Guided'),('random','Random'),('fixed','Fixed')]:
        c=next(r for r in hardware if int(r['coverage'])==coverage and r['rule']==rule and r['metric']=='calibration_error')
        p=next(r for r in hardware if int(r['coverage'])==coverage and r['rule']==rule and r['metric']=='pmd_at_ser_1pct')
        hw_body.append([f'$\\pm{coverage}^\\circ$',label,fmt(c['mean'],4),fmt(100*float(p['mean']),3)])
ablation += tex_table('Hardware-study means over three independent arrays, each with three optimization seeds. Active variants use 60 unique labels and fixed uses 10.',
    'tab:hardware-final','llrr',r'Coverage & Method & $e_d$ & PMD (\%)',hw_body)
hardware_pattern = re.compile(r'active_(weak|random|fixed)_hw(310[123])_s([012])_th(10|60)_cpu')
hardware_test = {}
for row in rows:
    match = hardware_pattern.search(Path(row['source']).name)
    if row['stage']=='stage1_validated' and match:
        hardware_test[(int(match[4]),match[1],int(match[2]),int(match[3]))] = float(row['pmd'])
array_means = {(coverage, rule):np.array([np.mean([hardware_test[(coverage,rule,h,s)] for s in range(3)])
    for h in [3101,3102,3103]]) for coverage in [10,60] for rule in ['weak','random','fixed']}
independent_hardware=[]
for coverage in [10,60]:
    for rule,label in [('weak','Guided'),('random','Random'),('fixed','Fixed')]:
        values=array_means[(coverage,rule)]
        differences=values-array_means[(coverage,'random')]
        independent_hardware.append([f'$\\pm{coverage}^\\circ$',label,
            f'${100*values.mean():.3f}\\pm{100*4.303*values.std(ddof=1)/np.sqrt(3):.3f}$',
            f'${100*differences.mean():+.3f}\\pm{100*4.303*differences.std(ddof=1)/np.sqrt(3):.3f}$'])
ablation += tex_table(r'Corrected independent-test hardware evaluation. Means and 95\% halfwidths use three array means, each averaging three optimization seeds. Differences are paired against random. Units are percent and percentage points (pp).',
    'tab:hardware-independent','llrr',r'Coverage & Method & PMD (\%) & Difference (pp)',independent_hardware)
ablation += r'''The corrected frozen-point hardware tests in Table~\ref{tab:hardware-independent} give guided PMD 8.923\% under narrow coverage and 10.390\% under broad coverage. Random selection gives 10.127\% and 9.266\%, and fixed labels give 8.674\% and 8.831\%. Guided-minus-random differences are $-1.203\pm2.530$ and $+1.124\pm4.939$ percentage points. These intervals also include zero, so the independent test retains the absence of an established guided advantage. Operating-point selection is completed on validation data before these test values are observed.
'''
ablation += r'''Under narrow coverage, guided-minus-random calibration error is $-0.0226\pm0.0512$ and PMD differs by $-1.363\pm2.116$ percentage points. Under broad coverage the differences are $+0.0375\pm0.1914$ and $+1.100\pm5.220$ percentage points. All four intervals include zero. Fixed labels have the lowest mean calibration error and PMD in both coverages. Their narrow PMD difference to random narrowly excludes zero, but that comparison is exploratory among several tests over three arrays.

The four information comparators match weak selection in all 72 query comparisons across coverage, hardware and optimization seeds. Independent hardware and broad angular support therefore do not separate these criteria in this study. The shared scene pools and three arrays limit population-level conclusions. The negative result concerns the tested single-target, plug-in-information setting and budgets.

\subsection{Numerical checks and runtime}
The original model checks gave relative errors $1.57\times10^{-5}$ for channel agreement, $8.05\times10^{-6}$ for information-model means, and $1.15\times10^{-9}$ for automatic versus finite-difference Jacobians. Translation and scale transformations preserve the fixed-design mean up to modeled gain phase within approximately $1.5\times10^{-14}$. These checks validate local calculations without establishing global identifiability under bounded sectors.

The corrected pilot's acquisition computation averages about 24.0 s per weak run, 26.2 s per D-optimal run, and 0.003 s per random run. The original laptop CPU pilot required 711--736 s of training per run and belongs to a different execution environment. All full-budget fresh runs used a local RTX 4090 with PyTorch 2.5.1+cu121, and saved optimizer checkpoints every 1,000 updates. Table~\ref{tab:runtime-final} gives accumulated training time, excluding downtime, validation and test jobs.
'''
runtime=[]
for n in allocations:
    seed=10 if n==0 else 42
    file=COHORT/'stage2_fraction_models'/f'fraction_sl{n}_total85000_seed{seed}.npz'
    with np.load(file) as z:
        config=json.loads(str(z['config_json']))
    runtime.append([f'{n:,}',seed,fmt(config['training_seconds']/3600,2)])
ablation += tex_table('Measured fresh-run training time. All models receive 85,000 updates and batch size 3,000.',
    'tab:runtime-final','rrr','Supervised updates & Seed & Training (h)',runtime)

discussion = r'''\section{Discussion and Limitations}
The experiments answer two related questions with different evidence. The reproduction and fraction sweep assess the effect of spending optimization updates on position supervision. The acquisition experiments assess whether selecting additional fixed-scene position references helps beyond continued use of existing labels. The archived SSL result motivates sparse supervision, while the fresh sweep documents performance changes over four seed-42 allocations under independent operating-point testing. Neither observation by itself establishes the benefit of choosing which fixed scenes to label.

The acquisition controls support a qualified null finding. Information-based criteria agree exactly on labels under all tested hardware/coverage comparisons. Fixed-label continuation is competitive, and guided-minus-random hardware intervals include zero. The information criterion therefore lacks demonstrated incremental value over its simpler information comparators in this setting. The diagnostic gives a concrete reason for the criterion agreement without claiming that every possible ISAC configuration has proportional label gains.

Evaluation changes materially affect the reported numbers. The original sensing and joint-frontier routines remain relevant for matching the published code. Their threshold reuse, independently sorted interpolation, shared-range batches and large sample sizes differ from the independently calibrated frozen-point evaluation. We report both with explicit labels. Differences between those protocols are not training gains. Likewise, conditional detected-target RMSE and unconditional localization RMSE describe different populations and should be interpreted with their accompanying error-tail statistics.

Several limitations constrain the final claims. The fresh sweep has one optimization seed per supervised allocation, with an additional differently seeded UL reference. There are only three independent hardware draws and one shared scene pool per coverage. SNR transfer uses fixed checkpoints and validation adaptation, rather than independent retraining at each SNR. Test intervals condition on trained weights and calibrated thresholds and omit variation in training, scene-pool selection and calibration sets. A complete budget for reference measurements, tuning labels and acquisition computation has not been estimated. The active setup assumes repeatable interrogation of fixed scenes with valid reference positions.

The held-out localization data confirm a heavy range-error tail even with known array parameters. That supports reporting median error and exact MSE contributions alongside conditional RMSE. Because the receiver selects a peak in a joint angle--delay map, the observed radial dominance does not prove that the calibration estimate has no influence on range selection. Causal receiver modifications and broader scene/hardware distributions would require additional experiments.

\section{Conclusion}
We completed full-budget SL, UL and sequential SSL reproduction, a supervised-allocation sweep, repaired operating-point evaluation, controlled acquisition ablations, hardware/coverage repetitions, SNR transfer and independent localization analysis. The archived 1,000-SL/84,000-UL model approaches full-SL detection under the reproduction protocol. In the fresh seed-42 sweep, 5,000 supervised updates give lower independent-test misdetection than the smaller completed allocations and detected-target RMSE of 1.649 m. The active-calibration extension yields a reproducible null result under the tested model, with identical information-based queries, wide hardware intervals and competitive fixed-label continuation.

The principal limitations are sparse trained-model repetition, only three hardware draws, shared scene pools and incomplete unique-reference cost accounting. Future work should replicate supervised allocations across seeds, vary acquisition information geometry, and evaluate receiver changes targeting the measured localization tail. These conclusions follow the completed evidence while retaining the distinction between reproduction, independent testing and acquisition-budget experiments.
'''
appendix = r'''\appendices
\section{Nuisance Elimination and Training Loop}
Partition the information into hardware $u$ and nuisance $z$ blocks. The regularized effective information is
\begin{equation}
 \F_{u\mid z}=\mathbf I_{uu}-\mathbf I_{uz}(\mathbf I_{zz}+\epsilon\mathbf D)^{-1}\mathbf I_{zu},
\end{equation}
where $\epsilon\mathbf D$ is a trace-scaled ridge. The unlabeled calculation eliminates angle, range and complex gain. The labeled calculation eliminates only complex gain, and the soft sector precision enters only the unlabeled angle block. Derivatives condition on the transmitted beam.

The loop initializes at nominal spacing, trains SL then UL, estimates candidate geometry/gain, computes information gains, selects a batch, reveals positions, and repeats training/evaluation. Figure~\ref{fig:workflow} shows the cycle. Validation metrics do not enter optimization gradients.
'''
workflow = next(m.group() for m in re.finditer(r'\\begin\{figure\}\[!htb\].*?\\end\{figure\}', old, re.S) if r'\label{fig:workflow}' in m.group())
appendix += workflow
appendix += r'''\section{Reproducibility and Artifact Access}
The final repository is \url{https://github.com/muneefo/DL_FINAL_REPORT}. It includes source, generated tables/figures, raw arrays, checkpoints and run logs. Git LFS stores the large ZIP archives. The imported baseline is \url{https://github.com/josemateosramos/SSLISAC}, associated with~\cite{ssl}. The team's earlier extension repository is \url{https://github.com/andrewliang-uwcs/867-deep-learning}. The original archive's precise upstream commit was not recorded, and should not be inferred from a later repository head. Original license and source attribution are retained.

The top-level \texttt{lib/}, \texttt{methods/} and \texttt{results/} preserve the reproduction workspace. Extended scripts live in \texttt{final-report-work/}. The completed cohorts are named \texttt{final\_stage\_20261108-142922}, \texttt{hardware\_study\_20261108-160301} and \texttt{course\_completion\_20261109-133839}. The completion state records successful jobs and final status. Hardware manifests include source hashes and environment captures. Fresh fraction checkpoints include learning rates, seeds, allocations, timing, model/optimizer state and CPU/CUDA random states.

The command \texttt{python reports/make\_submission\_tables.py} mechanically regenerates the saved main and ablation evidence. \texttt{plot\_completion\_results.py} regenerates the sweep and SNR graphics. \texttt{build\_final\_report.py} builds this report from manuscript components and saved results. Its bibliography is \texttt{refs.bib}, compiled with IEEEtran style. Full retraining is expensive and separate from table generation. The release contains the completed raw results so graders can verify values without retraining.

\section{Contribution and Assistance Disclosure}
The original team statement assigns Muneef leadership of baseline reproduction, mathematical methodology, extension planning and experimental analysis. Andrew contributed to proposal writing, related-work synthesis and integration. Mohammad contributed to the evaluation plan. Andrew and Mohammad also contributed to results discussion, and all members contributed to the midterm report. The team must verify the final individual allocation against its actual work and earlier repository history before submission. This imported repository snapshot does not reconstruct earlier author commits.

AI assistance supported code implementation, experiment orchestration, evaluation auditing, diagnostic reasoning, numerical summaries, plotting and manuscript drafting. This includes assistance beyond grammar correction or boilerplate. The course specification restricts AI generation of the core methodological contribution or analysis. The team must accurately disclose the scope, establish ownership and understanding of the scientific work, and clarify acceptability with the instructor. The present draft does not certify that those human review and policy checks have been completed.

\section{Data Use}
The experiments use synthetic single-target sensing and communication scenes generated by the supplied physical model. No human-subject, medical or restricted external dataset is used in the reported experiments. Scene seeds and checkpoint metadata identify the generated pools. Public release of synthetic artifacts is governed by the retained source license and the team's attribution obligations.
'''

bibliography = r'''@inproceedings{icc,
 author={Mateos-Ramos, J. M. and H{\"a}ger, C. and Keskin, M. F. and Le Magoarou, L. and Wymeersch, H.},
 title={Model-Driven End-to-End Learning for Integrated Sensing and Communication},
 booktitle={IEEE International Conference on Communications (ICC)}, year={2023}, pages={5695--5700}, doi={10.1109/ICC45041.2023.10278889}}
@article{multi,
 author={Mateos-Ramos, J. M. and H{\"a}ger, C. and Keskin, M. F. and Le Magoarou, L. and Wymeersch, H.},
 title={Model-Based End-to-End Learning for Multi-Target Integrated Sensing and Communication Under Hardware Impairments},
 journal={IEEE Transactions on Wireless Communications}, year={2025}, volume={24}, number={3}, pages={2574--2589}}
@inproceedings{ssl,
 author={Mateos-Ramos, J. M. and Chatelier, B. and H{\"a}ger, C. and Keskin, M. F. and Le Magoarou, L. and Wymeersch, H.},
 title={Semi-Supervised End-to-End Learning for Integrated Sensing and Communications},
 booktitle={IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN)}, year={2024}, pages={132--138}}
@article{mbdl,
 author={Shlezinger, N. and Whang, J. and Eldar, Y. C. and Dimakis, A. G.},
 title={Model-Based Deep Learning}, journal={Proceedings of the IEEE}, year={2023}, volume={111}, number={5}, pages={465--499}, doi={10.1109/JPROC.2023.3247480}}
@inproceedings{gain,
 author={Mateos-Ramos, J. M. and H{\"a}ger, C. and Keskin, M. F. and Le Magoarou, L. and Wymeersch, H.},
 title={Unsupervised Learning for Gain-Phase Impairment Calibration in {ISAC} Systems},
 booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, year={2025}}
@inproceedings{music,
 author={Chatelier, B. and Mateos-Ramos, J. M. and Corlay, V. and H{\"a}ger, C. and Crussi{\`e}re, M. and Wymeersch, H. and Le Magoarou, L.},
 title={Physically Parameterized Differentiable {MUSIC} for {DoA} Estimation with Uncalibrated Arrays},
 booktitle={IEEE International Conference on Communications (ICC)}, year={2025}, note={arXiv:2411.15144}}
@article{multiul,
 author={Mateos-Ramos, J. M. and Chatelier, B. and Le Magoarou, L. and Shlezinger, N. and Wymeersch, H. and H{\"a}ger, C.},
 title={Unsupervised End-to-End Array Calibration for Multi-Target Integrated Sensing and Communication},
 journal={arXiv preprint arXiv:2604.00806}, year={2026}}
@article{weiss,
 author={Weiss, A. J. and Friedlander, B.}, title={Array Shape Calibration Using Sources in Unknown Locations---A Maximum Likelihood Approach},
 journal={IEEE Transactions on Acoustics, Speech, and Signal Processing}, volume={37}, number={12}, year={1989}, pages={1958--1966}, doi={10.1109/29.45542}}
@article{cohn,
 author={Cohn, D. A. and Ghahramani, Z. and Jordan, M. I.}, title={Active Learning with Statistical Models},
 journal={Journal of Artificial Intelligence Research}, year={1996}, volume={4}, pages={129--145}, doi={10.1613/jair.295}}
@inproceedings{coreset,
 author={Sener, O. and Savarese, S.}, title={Active Learning for Convolutional Neural Networks: A Core-Set Approach},
 booktitle={International Conference on Learning Representations (ICLR)}, year={2018}, note={arXiv:1708.00489}}
@inproceedings{badge,
 author={Ash, J. T. and Zhang, C. and Krishnamurthy, A. and Langford, J. and Agarwal, A.},
 title={Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds},
 booktitle={International Conference on Learning Representations (ICLR)}, year={2020}, note={arXiv:1906.03671}}
@inproceedings{bait,
 author={Ash, J. T. and Goel, S. and Krishnamurthy, A. and Kakade, S.},
 title={Gone Fishing: Neural Active Learning with Fisher Embeddings},
 booktitle={Advances in Neural Information Processing Systems}, year={2021}, note={arXiv:2106.09675}}
@inproceedings{adam,
 author={Kingma, Diederik P. and Ba, Jimmy}, title={Adam: A Method for Stochastic Optimization},
 booktitle={International Conference on Learning Representations (ICLR)}, year={2015}, url={https://arxiv.org/abs/1412.6980}}
@inproceedings{pytorch,
 author={Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and K{\"o}pf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
 title={{PyTorch}: An Imperative Style, High-Performance Deep Learning Library},
 booktitle={Advances in Neural Information Processing Systems}, volume={32}, year={2019}, url={https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html}}
@article{wilson,
 author={Wilson, Edwin B.}, title={Probable Inference, the Law of Succession, and Statistical Inference},
 journal={Journal of the American Statistical Association}, year={1927}, volume={22}, number={158}, pages={209--212}, doi={10.1080/01621459.1927.10502953}}
'''
bibliography = re.sub(r'\\{2,}', lambda m:chr(92), bibliography)
document = preamble+'\n'+r'\begin{document}\maketitle'+'\n'+r'\begin{abstract}'+abstract+r'\end{abstract}'+'\n'
document += r'\begin{IEEEkeywords}ISAC, model-based learning, semi-supervised learning, active calibration, reproducibility.\end{IEEEkeywords}'+'\n'
document += intro+related+system+acquisition+protocol+results+ablation+discussion+appendix
document = document.replace(r'\texttt{python reports/make\_submission\_tables.py}',
                            r'\texttt{python} \path{reports/make_submission_tables.py}')
# Keep the main independent-test comparison as Table I for the course's
# single-command main-table requirement. Put reference configuration tables
# together in the appendix and place the archived comparison after Table I.
def take_table(label):
    global document
    matches = list(re.finditer(r'\\begin\{table\*?\}\[t\].*?\\end\{table\*?\}', document, re.S))
    match = next(m for m in matches if '\\label{'+label+'}' in m.group())
    value = match.group()
    document = document[:match.start()] + document[match.end():]
    return value

related_table = take_table('tab:related')
configuration_table = take_table('tab:settings-final')
archived_table = take_table('tab:legacy-final')
main_matches = list(re.finditer(r'\\begin\{table\*\}\[t\].*?\\end\{table\*\}', document, re.S))
main_match = next(m for m in main_matches if r'\label{tab:main-final}' in m.group())
document = document[:main_match.end()]+'\n'+archived_table+'\n'+document[main_match.end():]
document = document.replace(r'\section{Reproducibility and Artifact Access}',
    r'\section{Reference Tables}'+'\n'+configuration_table+'\n'+related_table+'\n'+
    r'\section{Reproducibility and Artifact Access}',1)
document = document.replace('Table~\\ref{tab:related} summarizes the reviewed works.',
                            'Appendix Table~\\ref{tab:related} summarizes the reviewed works.')
document += '\n'+r'\clearpage\raggedbottom\bibliographystyle{IEEEtran}\bibliography{refs}'+'\n'+r'\end{document}'+'\n'
# The conclusion belongs before the appendix.
document = document.replace(r'\appendices', discussion[discussion.index(r'\section{Conclusion}'):]+ '\n'+r'\appendices',1) if r'\section{Conclusion}' not in document else document
labels = re.findall(r'\\label\{([^}]+)\}', document)
refs = re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', document)
assert len(labels)==len(set(labels)), 'Duplicate labels'
assert set(refs)<=set(labels), set(refs)-set(labels)
assert len(abstract.split()) <= 250
assert len(re.findall(r'@(?:article|inproceedings)\{',bibliography)) >= 15
(OUT/'final_report.tex').write_text(document, encoding='utf-8')
(OUT/'refs.bib').write_text(bibliography, encoding='utf-8')
(OUT/'README.md').write_text('''# Final report draft

Compile `final_report.tex` using pdfLaTeX, BibTeX, then pdfLaTeX twice, or use
Tectonic with BibTeX support. The source includes inline PGFPlots data and TikZ
figures and uses `refs.bib` with IEEEtran bibliography style.

Rebuild from saved evidence with `py -3.12 reports/build_final_report.py`.
The draft includes factual result integration and the existing methodology.
The contribution and AI-assistance appendix requires team/instructor review.
''', encoding='utf-8')
print(f'Created {OUT}; abstract words={len(abstract.split())}, references=15')
