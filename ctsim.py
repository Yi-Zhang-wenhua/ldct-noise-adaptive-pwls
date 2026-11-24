import numpy as np
import scipy.sparse as sp

MU_W = 0.0193  # water, 1/mm


def hu2mu(hu):
    return MU_W * (1.0 + np.asarray(hu, float) / 1000.0)


def mu2hu(mu):
    return 1000.0 * (np.asarray(mu) - MU_W) / MU_W


class Geometry:
    def __init__(self, N=256, h=1.5, n_views=360, nd=None):
        self.N, self.h = N, h
        self.n_views = n_views
        self.nd = nd if nd is not None else int(np.ceil(N * np.sqrt(2))) + 3
        self.ds = h
        self.theta = np.arange(n_views) * np.pi / n_views
        c = (N - 1) / 2.0
        self.x = (np.arange(N) - c) * h
        self.y = (c - np.arange(N)) * h
        self.s = (np.arange(self.nd) - (self.nd - 1) / 2.0) * self.ds
        self.X, self.Y = np.meshgrid(self.x, self.y)


# phantoms: list of ellipses (value, x0, y0, a, b, angle in degrees), sizes in mm

def raster(ellipses, g, ss=4):
    N, h = g.N, g.h
    off = (np.arange(ss) + 0.5) / ss - 0.5
    img = np.zeros((N, N))
    for ox in off:
        for oy in off:
            X = g.X + ox * h
            Y = g.Y + oy * h
            for (v, x0, y0, a, b, phi) in ellipses:
                p = np.deg2rad(phi)
                xr = (X - x0) * np.cos(p) + (Y - y0) * np.sin(p)
                yr = -(X - x0) * np.sin(p) + (Y - y0) * np.cos(p)
                img += v * ((xr / a) ** 2 + (yr / b) ** 2 <= 1.0)
    return img / ss ** 2


def analytic_sinogram(ellipses, g, ss=4):
    off = ((np.arange(ss) + 0.5) / ss - 0.5) * g.ds
    th = g.theta[:, None]
    sino = np.zeros((g.n_views, g.nd))
    for o in off:
        s = g.s[None, :] + o
        for (v, x0, y0, a, b, phi) in ellipses:
            p = np.deg2rad(phi)
            r2 = (a * np.cos(th - p)) ** 2 + (b * np.sin(th - p)) ** 2
            sp_ = s - (x0 * np.cos(th) + y0 * np.sin(th))
            d = r2 - sp_ ** 2
            sino += np.where(d > 0, 2 * v * a * b * np.sqrt(np.maximum(d, 0)) / r2, 0.0)
    return sino / ss


def _E(hu_add, x0, y0, a, b, phi=0.0):
    return (MU_W * hu_add / 1000.0, x0, y0, a, b, phi)


def phantom_head():
    sc = 120.0
    E = [(hu2mu(1000), 0, 0, 0.69 * sc, 0.92 * sc, 0),
         (hu2mu(40) - hu2mu(1000), 0, -0.0184 * sc, 0.6624 * sc, 0.874 * sc, 0)]
    E.append(_E(-30, 0.22 * sc, 0, 0.11 * sc, 0.31 * sc, -18))
    E.append(_E(-30, -0.22 * sc, 0, 0.16 * sc, 0.41 * sc, 18))
    E.append(_E(20, 0, 0.35 * sc, 0.21 * sc, 0.25 * sc, 0))
    E.append(_E(20, 0, 0.1 * sc, 0.046 * sc, 0.046 * sc, 0))
    E.append(_E(20, 0, -0.1 * sc, 0.046 * sc, 0.046 * sc, 0))
    E.append(_E(20, -0.08 * sc, -0.605 * sc, 0.046 * sc, 0.023 * sc, 0))
    E.append(_E(20, 0, -0.605 * sc, 0.023 * sc, 0.023 * sc, 0))
    E.append(_E(20, 0.06 * sc, -0.605 * sc, 0.023 * sc, 0.046 * sc, 0))
    return E


def phantom_thorax():
    T = hu2mu(40)
    E = [(hu2mu(-100), 0, 0, 170, 120, 0),             # fat
         (T - hu2mu(-100), 0, 0, 160, 110, 0),          # soft tissue
         (hu2mu(-850) - T, -88, 10, 52, 85, 8),         # lungs
         (hu2mu(-850) - T, 92, 10, 50, 82, -8),
         (hu2mu(60) - T, 5, 5, 32, 40, 20),             # heart
         (hu2mu(150) - T, -15, -50, 12, 12, 0),         # aorta
         (hu2mu(700) - T, 0, -80, 22, 20, 0),           # vertebra
         (hu2mu(20) - hu2mu(700), 0, -80, 9, 8, 0),
         (hu2mu(700) - T, 0, 98, 14, 7, 0)]             # sternum
    for xs in (-1, 1):
        for (x0, y0) in ((140, -40), (150, 10), (128, 62), (100, -80), (90, 88)):
            E.append((hu2mu(800) - T, xs * x0, y0, 6, 6, 0))   # ribs
    E.append(_E(25, -40, 80, 6, 6))                      # lesions
    E.append(_E(25, 40, 80, 4, 4))
    E.append(_E(-25, 0, 60, 5, 5))
    E.append((hu2mu(30) - hu2mu(-850), -90, 40, 4, 4, 0))   # nodules
    E.append((hu2mu(30) - hu2mu(-850), 95, -20, 3, 3, 0))
    return E


def phantom_lowcontrast():
    E = [(hu2mu(0), 0, 0, 110, 110, 0)]
    diam = [16, 12, 10, 8, 6]
    for k, d in enumerate(diam):
        ang = np.deg2rad(90 + k * 36)
        E.append(_E(15, 60 * np.cos(ang), 60 * np.sin(ang), d / 2, d / 2))
    for k, d in enumerate(diam):
        ang = np.deg2rad(270 + k * 36)
        E.append(_E(40, 60 * np.cos(ang), 60 * np.sin(ang), d / 2, d / 2))
    E.append(_E(1200, -85, 0, 9, 9))
    E.append(_E(1200, 85, 0, 9, 9))
    E.append(_E(300, 0, 0, 18, 18))
    return E


# projectors

def system_matrix(g, dtype=np.float32):
    # pixel-driven, linear interpolation (used for FBP back-projection)
    N, nd = g.N, g.nd
    Xf, Yf = g.X.ravel(), g.Y.ravel()
    npx = N * N
    rows, cols, vals = [], [], []
    pix = np.arange(npx)
    scale = g.h ** 2 / g.ds
    for m, th in enumerate(g.theta):
        s = Xf * np.cos(th) + Yf * np.sin(th)
        t = s / g.ds + (nd - 1) / 2.0
        k0 = np.floor(t).astype(np.int64)
        w1 = t - k0
        for kk, ww in ((k0, 1 - w1), (k0 + 1, w1)):
            ok = (kk >= 0) & (kk < nd) & (ww > 0)
            rows.append(m * nd + kk[ok])
            cols.append(pix[ok])
            vals.append((scale * ww[ok]).astype(dtype))
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    return sp.csr_matrix((vals, (rows, cols)), shape=(g.n_views * nd, npx), dtype=dtype)


def _ramp_int(x):
    return np.where(x > 0, 0.5 * x * x, 0.0)


def system_matrix_sf(g, dtype=np.float32):
    # exact strip integral (trapezoid footprint), used for reconstruction
    N, nd, h, ds = g.N, g.nd, g.h, g.ds
    Xf, Yf = g.X.ravel(), g.Y.ravel()
    pix = np.arange(N * N)
    rows, cols, vals = [], [], []
    for m, th in enumerate(g.theta):
        a = max(0.5 * h * abs(np.cos(th)), 1e-3 * h)
        b = max(0.5 * h * abs(np.sin(th)), 1e-3 * h)

        def C(s):
            return (_ramp_int(s + a + b) - _ramp_int(s + a - b) - _ramp_int(s - a + b)
                    + _ramp_int(s - a - b)) / (4 * a * b)
        sc = Xf * np.cos(th) + Yf * np.sin(th)
        t = sc / ds + (nd - 1) / 2.0
        kc = np.round(t).astype(np.int64)
        for off in (-2, -1, 0, 1, 2):
            k = kc + off
            sk = (k - (nd - 1) / 2.0) * ds
            v = (h * h / ds) * (C(sk + ds / 2 - sc) - C(sk - ds / 2 - sc))
            ok = (k >= 0) & (k < nd) & (v > 1e-7)
            rows.append(m * nd + k[ok]); cols.append(pix[ok]); vals.append(v[ok].astype(dtype))
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    return sp.csr_matrix((vals, (rows, cols)), shape=(g.n_views * nd, N * N), dtype=dtype)


# noise

def simulate_counts(p, I0, sigma_e2, rng):
    ybar = I0 * np.exp(-p)
    return rng.poisson(ybar).astype(float) + rng.normal(0, np.sqrt(sigma_e2), size=p.shape)


def log_and_weights(y, I0, sigma_e2, ymin=1.0):
    yc = np.maximum(y, ymin)
    p = -np.log(yc / I0)
    w = yc ** 2 / (yc + sigma_e2)
    return p, w


# FBP

def ramp_kernel(nd, ds, window="hann"):
    npad = int(2 ** np.ceil(np.log2(2 * nd)))
    n = np.arange(-npad // 2, npad // 2)
    hk = np.zeros(npad)
    hk[n == 0] = 1.0 / (4 * ds ** 2)
    odd = (n % 2) == 1
    hk[odd] = -1.0 / (np.pi ** 2 * ds ** 2 * n[odd] ** 2)
    H = np.real(np.fft.fft(np.fft.ifftshift(hk))) * ds
    if window == "hann":
        H *= 0.5 * (1 + np.cos(2 * np.pi * np.fft.fftfreq(npad)))
    return H, npad


def filter_sino(sino, g, window="hann"):
    H, npad = ramp_kernel(g.nd, g.ds, window)
    S = np.fft.fft(sino, n=npad, axis=-1)
    return np.real(np.fft.ifft(S * H, axis=-1))[..., :g.nd]


def fbp(sino, g, A, window="hann"):
    q = filter_sino(sino, g, window)
    bp = A.T @ q.ravel().astype(np.float32)
    u = (np.pi / g.n_views) * (g.ds / g.h ** 2) * bp
    return u.reshape(g.N, g.N)


def fbp_noise_map(var_sino, g, A, A2, window="hann"):
    # predicted FBP variance, Eq. (5) in the paper
    H, npad = ramp_kernel(g.nd, g.ds, window)
    hk = np.real(np.fft.ifft(H))
    rho0 = np.sum(hk ** 2)
    rho1 = np.sum(hk * np.roll(hk, 1))
    v = var_sino.ravel().astype(np.float32)
    sc = g.ds / g.h ** 2
    S1 = sc * (A.T @ v)
    S2 = sc ** 2 * (A2.T @ v)
    var = (np.pi / g.n_views) ** 2 * (rho0 * S2 + rho1 * (S1 - S2))
    return var.reshape(g.N, g.N)
