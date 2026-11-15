"""Detection evaluation with separate null calibration, beam validation and test.

Keep legacy evaluation intact for reproduction. No interpolation or sorting of
unpaired metrics is used here. Every frame draws its own target range.
"""
import math
from types import SimpleNamespace
import numpy as np
import torch
from lib.functions import (createPrecoder, createCommonPrecoder, createMessages,
    sensingChannel, commChannel, sensingReceiver, createKappa, MLdecoder)


def wilson(errors, total, z=1.96):
    """Wilson interval for independent Bernoulli frames; NaN for empty strata."""
    if not total:
        return [float('nan'), float('nan')]
    p = errors / total
    denominator = 1 + z*z/total
    centre = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0., centre-half), min(1., centre+half)]


def null_threshold(metrics, alpha):
    """Finite-sample upper order statistic; detector uses strict greater-than."""
    values = np.asarray(metrics)
    if not 0 < alpha < 1 or not np.isfinite(values).all() or not values.size:
        raise ValueError('Invalid null calibration sample or false-alarm target')
    rank = math.ceil((len(values)+1)*(1-alpha))
    return float(np.sort(values)[rank-1]) if rank <= len(values) else float('inf')


class ValidatedEvaluator:
    def __init__(self, globals_dict, true_d, learned_d, sensing_snr=15., sector=(-40., -20.)):
        self.g = SimpleNamespace(**globals_dict)
        g = self.g
        self.true_d = torch.as_tensor(true_d, device=g.device, dtype=torch.float32).reshape(g.K, 1)
        self.d = torch.as_tensor(learned_d, device=g.device, dtype=torch.float32).reshape(g.K, 1)
        self.sigma = torch.sqrt(10**(sensing_snr/10)*g.N0/g.K)
        self.sector = list(sector)
        self.lo, self.hi = [torch.tensor(v*math.pi/180, device=g.device) for v in sector]
        torch.manual_seed(86001)
        f, _, _, self.A, self.b_angle = createPrecoder((self.lo+self.hi)/2,
            (self.lo+self.hi)/2, self.hi-self.lo, self.hi-self.lo, g.K, 1,
            d=self.d, lamb=g.lamb, Ngrid=g.Ngrid_angle, device=g.device)
        fc, _, _, _, _ = createPrecoder((g.theta_min_comm_test+g.theta_max_comm_test)/2,
            (g.theta_min_comm_test+g.theta_max_comm_test)/2,
            g.theta_max_comm_test-g.theta_min_comm_test,
            g.theta_max_comm_test-g.theta_min_comm_test, g.K, 1,
            d=self.d, lamb=g.lamb, Ngrid=g.Ngrid_angle, device=g.device)
        self.fr, self.fc = f[:, 0].reshape(1, -1), fc[:, 0].reshape(1, -1)
        self.b_range = ((g.range_grid >= g.range_min_sens_test) &
                        (g.range_grid <= g.range_max_sens_test)).reshape(1, 1, -1).to(torch.cfloat)

    @torch.no_grad()
    def collect(self, n, seed, rho, phi, batch=128, null_only=False):
        g = self.g
        torch.manual_seed(seed)
        output = {k: [] for k in ['metric', 'presence', 'true_angle', 'est_angle',
                                  'true_range', 'est_range', 'frame_ser']}
        for start in range(0, n, batch):
            b = min(batch, n-start)
            target = (torch.zeros((b, 1, 1), device=g.device) if null_only else
                      torch.randint(0, 2, (b, 1, 1), device=g.device).float())
            msg = createMessages(len(g.refConst), num=b*g.S, device=g.device)
            symbols = g.refConst[msg].reshape(b, g.S, 1)
            weight = torch.tensor(rho, dtype=torch.float32, device=g.device)
            phase = torch.tensor(phi, dtype=torch.float32, device=g.device)
            beam = createCommonPrecoder(self.fr, self.fc, weight, phase, b).reshape(b, g.K, 1)
            # Legacy scalar range limits share one range across a batch. Expand
            # explicitly so independent frames have independent target ranges.
            rmin = g.range_min_sens_test.expand(b, 1)
            rmax = g.range_max_sens_test.expand(b, 1)
            Y, theta, rng = sensingChannel(self.sigma, self.lo, self.hi, beam,
                symbols, rmin, rmax, g.N0, target, g.Delta_f, g.lamb, self.true_d, g.device)
            metric, angle, distance = sensingReceiver(Y, symbols, self.A, self.b_angle.expand(-1, b),
                self.b_range, g.range_grid, g.pixels_angle, g.pixels_range,
                g.S, g.Delta_f, d=self.d, lamb=g.lamb, device=g.device)
            if null_only:
                frame_ser = torch.zeros(b, device=g.device)
            else:
                received, comm_theta, beta = commChannel(g.sigma_vector_comm,
                    g.theta_min_comm_test, g.theta_max_comm_test, g.K, self.true_d,
                    g.lamb, beam, symbols, g.N0, g.device)
                kappa = createKappa(beta, comm_theta, g.K, self.true_d, g.lamb, beam, g.device)
                decoded = MLdecoder(received, kappa, g.refConst)
                frame_ser = (decoded.reshape(b, g.S) != msg.reshape(b, g.S)).float().mean(1)
            for key, value in dict(metric=metric[:, 0], presence=target,
                    true_angle=theta, est_angle=angle, true_range=rng,
                    est_range=distance, frame_ser=frame_ser).items():
                output[key].append(value.detach().cpu().numpy().reshape(-1))
        return {k: np.concatenate(v) for k, v in output.items()}

    @staticmethod
    def metrics(data, threshold):
        present = data['presence'].astype(bool)
        detected = data['metric'] > threshold
        misses = int(np.sum(present & ~detected))
        false = int(np.sum(~present & detected))
        n1, n0 = int(present.sum()), int((~present).sum())
        r, rh = data['true_range'], data['est_range']
        delta = data['true_angle'] - data['est_angle']
        e2 = (r-rh)**2 + 2*r*rh*(1-np.cos(delta))
        selected = present & detected
        return dict(pmd=misses/n1 if n1 else float('nan'),
            pfa=false/n0 if n0 else float('nan'), pmd_ci95=wilson(misses, n1),
            pfa_ci95=wilson(false, n0), n_present=n1, n_absent=n0,
            misses=misses, false_alarms=false, ser=float(data['frame_ser'].mean()),
            ser_frame_se=float(data['frame_ser'].std(ddof=1)/math.sqrt(len(present))),
            rmse_detected=float(np.sqrt(e2[selected].mean())) if selected.any() else float('nan'))

    def evaluate(self, n_cal=10000, n_validation=10000, n_test=20000,
                 beams=4, ser_limit=.01, alpha=.01):
        calibration_seed, validation_seed, test_seed = 87001, 88001, 89001
        calibration = self.collect(n_cal, calibration_seed, 0., 0., null_only=True)
        threshold = null_threshold(calibration['metric'], alpha)
        validation = []
        for rho in np.linspace(0, 1, beams):
            for phi in np.arange(beams)*2*np.pi/beams:
                sample = self.collect(n_validation, validation_seed, float(rho), float(phi))
                row = self.metrics(sample, threshold)
                row.update(rho=float(rho), phi=float(phi))
                validation.append(row)
        feasible = [i for i, row in enumerate(validation) if row['ser'] <= ser_limit]
        # No feasible point: report it, rather than select a beam using test data.
        if not feasible:
            return dict(status='no_validation_feasible_beam', threshold=threshold,
                validation=validation, selected_index=None), None
        selected = min(feasible, key=lambda i: (validation[i]['pmd'], validation[i]['ser'], i))
        row = validation[selected]
        sample = self.collect(n_test, test_seed, row['rho'], row['phi'])
        result = dict(status='ok', threshold=threshold, selected_index=selected,
            validation=validation, test=self.metrics(sample, threshold),
            n_cal=n_cal, n_validation=n_validation, n_test=n_test,
            calibration_seed=calibration_seed, validation_seed=validation_seed,
            test_seed=test_seed, sector=self.sector, beams_per_axis=beams,
            target_pfa=alpha, validation_ser_limit=ser_limit,
            protocol='independent_null_calibration_validation_beam_selection_and_test_v1')
        return result, sample
