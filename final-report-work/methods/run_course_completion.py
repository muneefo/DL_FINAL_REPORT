"""Sequential completion queue; stop on failures and keep resumable stage logs."""
from pathlib import Path
import subprocess
import sys
import os
import json
import shutil
import ctypes
from ctypes import wintypes
from datetime import datetime

root = Path(__file__).resolve().parents[1]
parent = root.parent
if len(sys.argv) > 1:
    cohort = Path(sys.argv[1]).resolve()
    state = json.loads((cohort/'state.json').read_text())
else:
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    cohort = root/'methods/results'/f'course_completion_{stamp}'
    cohort.mkdir(parents=True)
    state = dict(started=stamp, status='running', jobs=[],
        order=['validated evaluation', 'fixed-budget fraction sweep', 'SNR transfer tests',
               'independent localization', 'reproducibility and report evidence'],
        fraction_supervised_updates=[0, 100, 500, 1000, 5000, 10000, 85000],
        fraction_seeds=[10, 42], total_updates=85000,
        evaluation_calibration_frames=10000, evaluation_validation_frames=10000,
        evaluation_test_frames=20000, rubric_status='not supplied')
    for directory in ['lib', 'methods']:
        snapshot = cohort/'source_snapshot'/directory
        snapshot.mkdir(parents=True)
        for path in (root/directory).glob('*.py'):
            shutil.copy2(path, snapshot/path.name)
env = os.environ.copy()
env.update(ISAC_RUN_MODE='cpu', ISAC_COVERAGE='60', MPLBACKEND='Agg',
           PYTHONUNBUFFERED='1', MPLCONFIGDIR=str(root/'.mplconfig'))
env.pop('ISAC_HARDWARE_SEED', None)
env.pop('ISAC_RESULTS_DIR', None)
adopt_pid = int(sys.argv[2]) if len(sys.argv) > 2 else None
adopt_job = state.get('current_job') if adopt_pid else None
plan_file = parent/'reports/completion_priority_plan.json'
if plan_file.exists():
    priority_plan = json.loads(plan_file.read_text())
    planned_runs = [(int(r['seed']), int(r['supervised'])) for r in priority_plan['runs']]
    state.update(deadline=priority_plan['deadline'], priority_plan=priority_plan,
                 rubric_status='supplied; see reports/COURSE_COMPLIANCE.md')
else:
    planned_runs = [(seed, n) for seed in state['fraction_seeds'] for n in state['fraction_supervised_updates']]


def wait_for_existing_training(pid):
    """Adopt only the explicitly identified live training process on Windows."""
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x100000 | 0x1000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        while True:
            result = kernel.WaitForSingleObject(handle, 1000)
            if result == 0:
                break
            if result != 258:
                raise ctypes.WinError(ctypes.get_last_error())
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return code.value
    finally:
        kernel.CloseHandle(handle)


def save():
    (cohort/'state.json').write_text(json.dumps(state, indent=2))


def job(name, script, arguments, overrides=None):
    previous = next((j for j in reversed(state['jobs']) if j['name'] == name), None)
    if previous and previous.get('exit_code') == 0:
        return
    record = dict(name=name, script=str(script), args=[str(a) for a in arguments], started=datetime.now().isoformat())
    state['jobs'].append(record)
    state['current_job'] = name
    state['status'] = 'running'
    save()
    child_env = env.copy()
    child_env.update(overrides or {})
    if adopt_pid and name == adopt_job:
        record['adopted_training_pid'] = adopt_pid
        save()
        code = wait_for_existing_training(adopt_pid)
    else:
        with (cohort/f'{name}.log').open('a') as stream:
            result = subprocess.run([sys.executable, '-u', str(script), *map(str, arguments)],
                                    cwd=root, env=child_env, stdout=stream, stderr=subprocess.STDOUT)
        code = result.returncode
    record.update(exit_code=code, finished=datetime.now().isoformat())
    if code:
        state['status'] = 'failed'
        save()
        raise SystemExit(code)
    save()


save()
# Use exact established cohorts; do not mix historical pilot archives.
results = root/'methods/results'
active = [p for p in results.glob('active_*_th10_cpu_20261108-*.npz')
          if '20261108-142922' <= p.stem[-15:] <= '20261108-145259']
hardware = list((results/'hardware_study_20261108-160301').glob('hw*/active_*.npz'))
assert len(active) == 50 and len(hardware) == 126, (len(active), len(hardware))
parent_models = [parent/'results'/name for name in ['SL_85000.npz', 'UL_85000.npz', 'SSL_SL_UL_1000.npz']]
job('stage1_controls', root/'methods/prepare_known_controls.py', ['--output', cohort/'known_controls'])
controls = sorted((cohort/'known_controls').glob('*.npz'))
models = sorted(active+hardware)+parent_models+controls
job('stage1_validated', root/'methods/evaluate_validated.py',
    ['--output', cohort/'stage1_validated', *models])
job('stage1_summary', root/'methods/summarize_completion.py', [cohort])

fractions = cohort/'stage2_fraction_models'
fractions.mkdir(exist_ok=True)
for seed, supervised in planned_runs:
        job(f'stage2_train_seed{seed}_sl{supervised}', root/'methods/train_fraction.py',
            ['--supervised', supervised, '--total', 85000, '--seed', seed, '--output', fractions],
            dict(ISAC_RUN_MODE='paper', ISAC_COVERAGE='60', ISAC_SEED=str(seed)))
        checkpoint = fractions/f'fraction_sl{supervised}_total85000_seed{seed}.npz'
        job(f'stage2_evaluate_seed{seed}_sl{supervised}', root/'methods/evaluate_validated.py',
            ['--output', cohort/'stage2_sweep_evaluation', checkpoint])
job('stage2_summary', root/'methods/summarize_completion.py', [cohort])

# Evaluate trained models at changed sensing SNR; this is transfer, not retraining.
selected = [p for p in hardware if any(p.name.startswith(f'active_{r}_hw') for r in ['weak', 'random', 'fixed'])]
selected += parent_models + sorted(fractions.glob('fraction_*.npz'))
job('stage3_snr', root/'methods/evaluate_validated.py',
    ['--output', cohort/'stage3_snr', '--snrs', 5, 10, 20, 25, '--', *selected, *controls])
job('stage4_localization', root/'methods/localization_test.py',
    ['--output', cohort/'stage4_localization', '--samples', 5000, *models, *sorted(fractions.glob('fraction_*.npz'))])
job('stage4_summary', root/'methods/summarize_completion.py', [cohort])
job('stage5_package', parent/'reports/package_completion.py', [cohort])
state.update(status='experiments_complete', finished=datetime.now().isoformat(),
             next='Compile/inspect final report, prepare and verify presentation, and check supplied course rubric')
save()
