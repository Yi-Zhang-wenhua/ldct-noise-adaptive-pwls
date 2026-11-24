# extra studies. usage: python exp_extra.py conv|euler|nr|design|hybrid
import sys, json
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from exp_main import *


def setup(name, I0, seed):
    x = np.load(f'data/gt_{name}.npy'); p = np.load(f'data/sino_{name}.npy'); M = masks(x)
    ph, w, u_fbp, sig, cert = realise(p, I0, seed)
    s = np.median((A.T @ (w.ravel() * A1)).reshape(N, N)[M[0]])
    return x, M, ph, w, u_fbp, sig, cert, s


def batch(ph, w, u_fbp, nb):
    P = np.repeat(ph.ravel()[:, None], nb, 1).astype(np.float32)
    W = np.repeat(w.ravel()[:, None], nb, 1).astype(np.float32)
    U0 = np.repeat(np.maximum(u_fbp, 0)[..., None], nb, 2).astype(np.float32)
    return P, W, U0


def conv_study():
    x, M, ph, w, u_fbp, sig, cert, s = setup('thorax', 1e4, 4242)
    k = np.float32(5 * HU); sref = np.median(sig[M[0]])
    beta = (0.2 * s * k * sref / sig).astype(np.float32)[..., None]
    pr = Prior('charb', beta=beta, k=k)
    P, W, U0 = batch(ph, w, u_fbp, 1)
    op = LinearOperator((N * N, N * N), matvec=lambda v: A.T @ (w.ravel() * (A @ v.astype(np.float32))), dtype=np.float32)
    Ldata = float(eigsh(op, k=1, which='LM', return_eigenvectors=False, tol=1e-3)[0])
    L = Ldata + 8 * float(np.max(beta / k))
    out = {'Ldata': Ldata, 'L': L}
    uref, _, D = pwls(A, P, W, pr, U0, n_iter=3000, restart=True, A1=A1)   # reference minimiser
    r = (A @ uref.reshape(-1, 1)) - P
    Fstar = float(0.5 * np.sum(W * r ** 2) + pr.value(uref)[0])
    out['Fstar'] = Fstar
    out['D_norm_u0'] = float(np.sum(D * (U0 - uref) ** 2))
    runs = {'GD (1/L)': dict(momentum=False, step=1.0 / L), 'SQS': dict(momentum=False),
            'SQS-Nesterov': dict(momentum=True), 'SQS-Nesterov-restart': dict(momentum=True, restart=True)}
    for nm, kw in runs.items():
        u, h, _ = pwls(A, P, W, pr, U0, n_iter=400, track=True, A1=A1, U_ref=uref, **kw)
        out[nm] = dict(gap=[float(r['obj'][0] - Fstar) for r in h],
                       dist_hu=[float(r['rmse'][0] / HU) for r in h])
        print(nm, out[nm]['gap'][-1], out[nm]['dist_hu'][-1], flush=True)
    json.dump(out, open('results/res_conv.json', 'w'))


def euler_study():
    # explicit scheme on the quadratic model, start perturbed along the top eigenvector
    x, M, ph, w, u_fbp, sig, cert, s = setup('thorax', 1e4, 4242)
    lam = np.float32(0.01 * s)

    def H(v):
        v = v.astype(np.float32)
        return A.T @ (w.ravel() * (A @ v)) + lam * gradT(*grad(v.reshape(N, N, 1)))[..., 0].ravel()
    val, vec = eigsh(LinearOperator((N * N, N * N), matvec=H, dtype=np.float32), k=1, which='LM', tol=1e-6)
    L = float(val[0]); v = vec[:, 0].reshape(N, N, 1).astype(np.float32)
    P, W, U0 = batch(ph, w, u_fbp, 1)
    U0 = U0 + 1e-3 * MU_W * v / np.abs(v).max()
    out = {'L_exact': L}
    for fac in [1.0, 1.9, 2.05, 2.5]:
        u, h, _ = pwls(A, P, W, Prior('quad', beta=lam), U0, n_iter=150, momentum=False,
                       step=fac / L, track=True, A1=A1, nonneg=False)
        out[str(fac)] = [float(r['obj'][0]) for r in h]
        print(fac, out[str(fac)][0], out[str(fac)][-1], flush=True)
    json.dump(out, open('results/res_euler.json', 'w'))


def nr_study():
    # noise-resolution trade-off, LC phantom
    x, M, *_ = setup('lc', 1e4, 1)
    p = np.load('data/sino_lc.npy')
    X, Y = g.X, g.Y
    roi_near = np.hypot(X - 85, Y - 22) < 7
    roi_far = np.hypot(X - 0, Y - 30) < 7
    rad = np.hypot(X, Y); bins = np.arange(8, 28, 0.5)
    lams = [0.1, 0.2, 0.5, 1.0, 2.0]
    out = {}
    for m in ['PWLS-U', 'PWLS-CB', 'PWLS-NA']:
        ims = []
        for sd in range(6):
            ph, w, u_fbp, sig, cert = realise(p, 1e4, 5000 + sd)
            ims.append(pwls_family(ph, w, u_fbp, sig, cert, M[0], [(m, l, 5.0) for l in lams]))
        res = []
        for j, l in enumerate(lams):
            stack = np.stack([ims[i][j] for i in range(6)])
            mean = stack.mean(0); std = stack.std(0, ddof=1) / HU
            # edge spread of the central +300 HU disc
            prof = np.array([mean[(rad >= a) & (rad < a + 0.5)].mean() for a in bins]) / HU
            f = (prof - prof[-4:].mean()) / (prof[:4].mean() - prof[-4:].mean())

            def cross(level):
                i = np.where(f < level)[0][0]
                return bins[i - 1] + (f[i - 1] - level) / (f[i - 1] - f[i]) * 0.5
            res.append(dict(lam=l, noise_near=float(std[roi_near].mean()), noise_far=float(std[roi_far].mean()),
                            esf_10_90_mm=float(cross(0.1) - cross(0.9)),
                            rmse=float(np.mean([metrics(ims[i][j], x, M)['rmse'] for i in range(6)]))))
            print(m, res[-1], flush=True)
        out[m] = res
    json.dump(out, open('results/res_nr.json', 'w'))


def design_study():
    # weight map and exponent, separate realisation
    out = {}
    for name in ['thorax', 'lc', 'head']:
        x, M, ph, w, u_fbp, sig, cert, s = setup(name, 1e4, 777)
        sref = np.median(sig[M[0]]); cref = np.median(cert[M[0]])
        maps = {f'fbp_g{gm}': (sref / sig) ** gm for gm in [0, 1, 2, 3]}
        maps.update({f'cert_e{e}': (cert / cref) ** e for e in [0.5, 1.0, 1.5]})
        cfg = [(l, kk) for l in [0.2, 0.5, 1, 2, 5, 10] for kk in [2.5, 5.0]]
        kv = np.array([c[1] * HU for c in cfg], np.float32)
        for mn, eta in maps.items():
            beta = eta[..., None].astype(np.float32) * (np.array([c[0] * s for c in cfg], np.float32) * kv)
            res = []
            for b0 in range(0, len(cfg), 12):
                sl = slice(b0, b0 + 12); nb = len(cfg[sl])
                P, W, U0 = batch(ph, w, u_fbp, nb)
                u, _, _ = pwls(A, P, W, Prior('charb', beta=beta[..., sl], k=kv[sl]), U0, n_iter=NIT, restart=True, A1=A1)
                res += [dict(lam=cfg[b0 + j][0], k=cfg[b0 + j][1], **metrics(u[..., j], x, M)) for j in range(nb)]
            out[f'{name}_{mn}'] = res
            b = min(res, key=lambda r: r['rmse'])
            print(name, mn, b['lam'], b['k'], round(b['rmse'], 2), flush=True)
    json.dump(out, open('results/res_design.json', 'w'))


def hybrid_study():
    # certainty weight with threshold k = kappa * sigma
    out = {}
    for name in ['thorax', 'lc', 'head']:
        x, M, ph, w, u_fbp, sig, cert, s = setup(name, 1e4, 777)
        cref = np.median(cert[M[0]])
        cfg = [(l, kp) for l in [0.2, 0.5, 1, 2, 5, 10] for kp in [0.125, 0.25, 0.5, 1.0]]
        res = []
        for b0 in range(0, len(cfg), 12):
            cf = cfg[b0:b0 + 12]; nb = len(cf)
            kk = np.stack([kp * sig for _, kp in cf], -1).astype(np.float32)
            beta = np.stack([l * s * kp * sig * cert / cref for l, kp in cf], -1).astype(np.float32)
            P, W, U0 = batch(ph, w, u_fbp, nb)
            u, _, _ = pwls(A, P, W, Prior('charb', beta=beta, k=kk), U0, n_iter=NIT, restart=True, A1=A1)
            res += [dict(lam=cf[j][0], kappa=cf[j][1], **metrics(u[..., j], x, M)) for j in range(nb)]
        out[name] = res
        b = min(res, key=lambda r: r['rmse'])
        print(name, b['lam'], b['kappa'], round(b['rmse'], 2), flush=True)
    json.dump(out, open('results/res_hybrid.json', 'w'))


if __name__ == '__main__':
    {'conv': conv_study, 'euler': euler_study, 'nr': nr_study,
     'design': design_study, 'hybrid': hybrid_study}[sys.argv[1]]()
