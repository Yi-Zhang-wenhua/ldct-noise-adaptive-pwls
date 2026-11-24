# main comparison: tune on one realisation, test on six paired realisations
# usage: python exp_main.py WORKER_ID N_WORKERS
import os, sys, json, time
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import binary_dilation
from skimage.metrics import structural_similarity
from skimage.restoration import denoise_tv_chambolle
import bm3d
from ctsim import *
from recon import *

g = Geometry(); N = g.N
Al = sp.load_npz('data/A.npz'); Al2 = Al.multiply(Al).tocsr()
A = sp.load_npz('data/Asf.npz'); Asf2 = A.multiply(A).tocsr()
A1 = A @ np.ones(N * N, np.float32)
colsq = np.asarray(Asf2.sum(axis=0)).ravel()
HU = MU_W / 1000.0
SE2 = 10.0
NIT = int(os.environ.get('NIT', 200))
TEST_SEEDS = list(range(6))

LAM = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]; KS = [2.5, 5.0, 10.0]
LAM_Q = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
LAM_TV = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
PM_K = [10, 20, 40, 80, 160]; PM_N = [10, 30, 60, 100]
TV_W = [h * HU for h in [5, 10, 20, 40, 80, 160, 320]]
BM_S = [h * HU for h in [10, 20, 30, 45, 60, 80, 120]]


def masks(x):
    body = x > hu2mu(-990)
    gx, gy = np.gradient(x)
    edge = binary_dilation(np.hypot(gx, gy) > 50 * HU, iterations=2) & body
    return body, edge, body & ~edge


def metrics(u, x, M):
    body, edge, smooth = M
    e = (u - x) / HU
    rng_ = x[body].max() - x[body].min()
    ssim_map = structural_similarity(x, u, data_range=rng_, full=True)[1]
    mse = np.mean((u - x)[body] ** 2)
    return dict(rmse=float(np.sqrt(np.mean(e[body] ** 2))),
                rmse_smooth=float(np.sqrt(np.mean(e[smooth] ** 2))),
                rmse_edge=float(np.sqrt(np.mean(e[edge] ** 2))),
                ssim=float(np.mean(ssim_map[body])),
                psnr=float(10 * np.log10(rng_ ** 2 / mse)))


def realise(p, I0, seed):
    rng = np.random.default_rng(seed)
    y = simulate_counts(p, I0, SE2, rng)
    ph, w = log_and_weights(y, I0, SE2)
    u_fbp = fbp(ph, g, Al)
    sig = np.sqrt(fbp_noise_map(1.0 / w, g, Al, Al2))
    cert = (Asf2.T @ w.ravel()) / np.maximum(colsq, 1e-12)
    return ph, w, u_fbp, sig, cert.reshape(N, N)


def pwls_family(ph, w, u_fbp, sig, cert, body, configs):
    # configs: list of (method, lam, k)
    s = np.median((A.T @ (w.ravel() * A1)).reshape(N, N)[body])
    sref = np.median(sig[body]); cref = np.median(cert[body])
    out = []
    for b0 in range(0, len(configs), 16):
        cf = configs[b0:b0 + 16]; B = len(cf)
        kind = 'quad' if cf[0][0] == 'PWLS-Q' else 'charb'
        beta = np.zeros((N, N, B), np.float32); kk = np.zeros(B, np.float32)
        for j, (m, lam, k) in enumerate(cf):
            if m == 'PWLS-Q':
                beta[..., j] = lam * s
            elif m == 'PWLS-TV':
                kk[j] = 1.0 * HU; beta[..., j] = lam * s * 10 * HU
            else:
                kk[j] = k * HU
                eta = {'PWLS-U': 1.0, 'PWLS-CB': cert / cref, 'PWLS-NA': sref / sig}[m]
                beta[..., j] = lam * s * kk[j] * eta
        prior = Prior('quad', beta=beta) if kind == 'quad' else Prior('charb', beta=beta, k=kk)
        P = np.repeat(ph.ravel()[:, None], B, 1).astype(np.float32)
        W = np.repeat(w.ravel()[:, None], B, 1).astype(np.float32)
        U0 = np.repeat(np.maximum(u_fbp, 0)[..., None], B, 2).astype(np.float32)
        u, _, _ = pwls(A, P, W, prior, U0, n_iter=NIT, restart=True, A1=A1)
        out += [u[..., j] for j in range(B)]
    return out


PWLS_GRIDS = {'PWLS-Q': [('PWLS-Q', l, None) for l in LAM_Q],
              'PWLS-TV': [('PWLS-TV', l, None) for l in LAM_TV]}
for m in ['PWLS-U', 'PWLS-CB', 'PWLS-NA']:
    PWLS_GRIDS[m] = [(m, l, k) for l in LAM for k in KS]


def image_methods(u_fbp, params=None):
    out = {}
    pm = [(K, n) for K in PM_K for n in PM_N] if params is None else [params['FBP+PM']]
    out['FBP+PM'] = [(q, perona_malik(u_fbp, q[0] * HU, q[1])) for q in pm]
    tv = TV_W if params is None else [params['FBP+TV']]
    out['FBP+TV'] = [(q, denoise_tv_chambolle(u_fbp, weight=q)) for q in tv]
    bs = BM_S if params is None else [params['FBP+BM3D']]
    out['FBP+BM3D'] = [(q, bm3d.bm3d(u_fbp, sigma_psd=q)) for q in bs]
    return out


def run_case(name, I0):
    tag = f'{name}_{int(I0)}'
    x = np.load(f'data/gt_{name}.npy'); p = np.load(f'data/sino_{name}.npy'); M = masks(x)
    t0 = time.time()

    # tuning
    ph, w, u_fbp, sig, cert = realise(p, I0, 1000 + int(I0) + len(name))
    best = {}; tune_log = {}
    for m, lst in image_methods(u_fbp).items():
        sc = [(q, metrics(u, x, M)['rmse']) for q, u in lst]
        best[m] = min(sc, key=lambda t: t[1])[0]; tune_log[m] = sc
    for m, grid in PWLS_GRIDS.items():
        ims = pwls_family(ph, w, u_fbp, sig, cert, M[0], grid)
        sc = [(c, metrics(u, x, M)['rmse']) for c, u in zip(grid, ims)]
        best[m] = min(sc, key=lambda t: t[1])[0]; tune_log[m] = sc
    print(tag, 'tuned', best, round(time.time() - t0), flush=True)

    # test
    res = {m: [] for m in ['FBP', 'FBP+PM', 'FBP+TV', 'FBP+BM3D'] + list(PWLS_GRIDS)}
    for sd in TEST_SEEDS:
        ph, w, u_fbp, sig, cert = realise(p, I0, 10 * sd + int(I0) % 997 + 7 * len(name))
        imgs = {'FBP': u_fbp}
        for m, lst in image_methods(u_fbp, best).items():
            imgs[m] = lst[0][1]
        imgs['PWLS-Q'] = pwls_family(ph, w, u_fbp, sig, cert, M[0], [best['PWLS-Q']])[0]
        rest = ['PWLS-TV', 'PWLS-U', 'PWLS-CB', 'PWLS-NA']
        for m, u in zip(rest, pwls_family(ph, w, u_fbp, sig, cert, M[0], [best[m] for m in rest])):
            imgs[m] = u
        for m, u in imgs.items():
            res[m].append(metrics(u, x, M))
        if sd == 0:
            np.savez_compressed(f'results/img_{tag}.npz', gt=x, sig=sig, cert=cert,
                                **{m.replace('+', '_').replace('-', '_'): u for m, u in imgs.items()})
    json.dump(dict(best=best, tune=tune_log, test=res), open(f'results/res_main_{tag}.json', 'w'), default=str)
    print(tag, 'done', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    jobs = [(n, float(d)) for n in ['thorax', 'lc', 'head'] for d in [2e4, 1e4, 5e3]]
    wid, nw = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    for j, (n, d) in enumerate(jobs):
        if j % nw == wid and not os.path.exists(f'results/res_main_{n}_{int(d)}.json'):
            run_case(n, d)
