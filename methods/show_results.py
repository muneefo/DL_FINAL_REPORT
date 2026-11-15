"""Plot saved ISAC results without running training."""
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.plot-deps'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def pareto_front(x, y):
    valid = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    keep = [i for i in valid if not any(
        x[j] <= x[i] and y[j] <= y[i]
        and (x[j] < x[i] or y[j] < y[i]) for j in valid)]
    return np.array(sorted(keep, key=lambda i: x[i]), dtype=int)


def main():
    output = ROOT / 'results' / 'plots'
    output.mkdir(exist_ok=True)
    runs = []
    for filename, name in [
        ('baseline_unknown_impairments.npz', 'Baseline (unknown impairments)'),
        ('SL_85000.npz', 'SL'), ('UL_85000.npz', 'UL'),
        ('SSL_SL_UL_1000.npz', 'SSL: SL then UL'),
    ]:
        with np.load(ROOT / 'results' / filename) as saved:
            data = {key: saved[key] for key in saved.files}
        ns = len(data.get('loss_super', []))
        nu = len(data.get('loss_unsuper', []))
        if ns or nu:
            assert ns + nu == 85000, f'{filename}: expected 85,000 updates'
            assert data['num_iterations'][-1] == 85000
            print(f"Final sensing: PMD={1-data['pd_inter'][-1]:.6f}, "
                  f"position RMSE={data['rmse_pos_inter'][-1]:.6f} m, "
                  f"PFA={data['pfa_inter'][-1]:.6f}")
        label = name + (f' ({ns:,} SL + {nu:,} UL)' if ns or nu else '')
        runs.append((label, data))
        print(f'{filename}: {ns:,} supervised, {nu:,} unsupervised updates')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    for index, (label, data) in enumerate(runs):
        color = f'C{index}'
        if 'num_iterations' in data:
            iterations = data['num_iterations']
            axes[0, 0].semilogy(iterations, 1 - data['pd_inter'],
                                '.-', color=color, label=label)
            axes[0, 1].plot(iterations, data['rmse_pos_inter'],
                            '.-', color=color, label=label)
        ser = data['ser_isac']
        for axis, metric in [(axes[1, 0], 1 - data['pd_isac']),
                             (axes[1, 1], data['rmse_pos_isac'])]:
            front = pareto_front(ser, metric)
            axis.scatter(ser, metric, color=color, alpha=0.14, s=15)
            axis.plot(ser[front], metric[front], 'o-', color=color,
                      markersize=4, label=label)
            axis.set_xscale('log')
    axes[0, 0].set(title='Sensing detection during training',
                    xlabel='Training updates', ylabel='Misdetection probability')
    axes[0, 1].set(title='Position accuracy during training',
                    xlabel='Training updates', ylabel='Position RMSE [m]')
    axes[1, 0].set(title='ISAC detection versus communication',
                    xlabel='Symbol error rate', ylabel='Misdetection probability',
                    yscale='log')
    axes[1, 1].set(title='ISAC positioning versus communication',
                    xlabel='Symbol error rate', ylabel='Position RMSE [m]')
    for axis in axes.flat:
        axis.grid(True, which='both', alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle('ISAC comparison: 85,000 training updates per learned method', fontsize=15)
    fig.savefig(output / 'results_comparison.png', dpi=180)
    fig.savefig(output / 'results_comparison.svg')
    plt.close(fig)
    print(f'Plots saved to {output}')


if __name__ == '__main__':
    main()
