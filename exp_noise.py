# checks the FBP noise formula against Monte Carlo
import json
import numpy as np
import scipy.sparse as sp
from ctsim import *

g = Geometry(); A = sp.load_npz('data/A.npz'); A2 = A.multiply(A).tocsr()
se2 = 10.0; R = 300
out = {}
for name in ['thorax', 'lc']:
    x = np.load(f'data/gt_{name}.npy'); p = np.load(f'data/sino_{name}.npy')
    body = x > hu2mu(-990)
    for I0 in [2e4, 5e3]:
        rng = np.random.default_rng(123)
        ybar = I0 * np.exp(-p)
        var_true = (ybar + se2) / ybar ** 2
        s1 = np.zeros_like(p); s2 = np.zeros_like(p)
        U1 = np.zeros((g.N, g.N)); U2 = np.zeros((g.N, g.N))
        w_first = None
        for r in range(R):
            y = simulate_counts(p, I0, se2, rng)
            ph, w = log_and_weights(y, I0, se2)
            if w_first is None:
                w_first = w
            s1 += ph; s2 += ph ** 2
            u = fbp(ph, g, A); U1 += u; U2 += u ** 2
        vs_emp = s2 / R - (s1 / R) ** 2
        vi_emp = U2 / R - (U1 / R) ** 2
        vi_pred = fbp_noise_map(var_true, g, A, A2)
        vi_est = fbp_noise_map(1.0 / w_first, g, A, A2)
        bins = [5, 20, 50, 200, 1000, 1e5]
        sino_rel = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (ybar >= lo) & (ybar < hi)
            if m.sum() > 0:
                sino_rel.append([lo, hi, int(m.sum()), float(np.median(vs_emp[m] / var_true[m]))])
        se, sp_, ss = np.sqrt(vi_emp[body]), np.sqrt(vi_pred[body]), np.sqrt(vi_est[body])
        key = f'{name}_{int(I0)}'
        out[key] = dict(corr_oracle=float(np.corrcoef(se, sp_)[0, 1]), corr_est=float(np.corrcoef(se, ss)[0, 1]),
                        medrel_oracle=float(np.median(np.abs(sp_ - se) / se)),
                        medrel_est=float(np.median(np.abs(ss - se) / se)),
                        mc_rel_se=1.0 / np.sqrt(2 * (R - 1)), sino_ratio=sino_rel,
                        std_hu_emp_pct=list(np.percentile(se / MU_W * 1000, [5, 50, 95])),
                        std_hu_pred_pct=list(np.percentile(sp_ / MU_W * 1000, [5, 50, 95])))
        np.savez(f'results/res_noise_{key}.npz', emp=np.sqrt(vi_emp), pred=np.sqrt(vi_pred), est=np.sqrt(vi_est))
        print(key, json.dumps(out[key]))
json.dump(out, open('results/res_noise.json', 'w'), indent=1)
