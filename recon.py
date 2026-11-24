import numpy as np


def grad(u):
    dx = np.zeros_like(u); dy = np.zeros_like(u)
    dx[:, :-1] = u[:, 1:] - u[:, :-1]
    dy[:-1, :] = u[1:, :] - u[:-1, :]
    return dx, dy


def gradT(fx, fy):
    out = np.zeros_like(fx)
    out[:, 1:] += fx[:, :-1]
    out[:, :-1] -= fx[:, :-1]
    out[1:, :] += fy[:-1, :]
    out[:-1, :] -= fy[:-1, :]
    return out


def reg_majorizer(cmax):
    D = np.zeros_like(cmax)
    cx = cmax.copy(); cx[:, -1] = 0
    cy = cmax.copy(); cy[-1, :] = 0
    D += 2 * cx; D[:, 1:] += 2 * cx[:, :-1]
    D += 2 * cy; D[1:, :] += 2 * cy[:-1, :]
    return D


class Prior:
    # 'charb': beta * (sqrt(k^2 + t^2) - k),  'quad': beta * t^2 / 2
    def __init__(self, kind, beta, k=None):
        self.kind, self.beta, self.k = kind, beta, k

    def value(self, u):
        dx, dy = grad(u); t2 = dx ** 2 + dy ** 2
        if self.kind == 'quad':
            return np.sum(self.beta * 0.5 * t2, axis=(0, 1))
        return np.sum(self.beta * (np.sqrt(self.k ** 2 + t2) - self.k), axis=(0, 1))

    def gradient(self, u):
        dx, dy = grad(u)
        if self.kind == 'quad':
            c = self.beta * np.ones_like(u)
        else:
            c = self.beta / np.sqrt(self.k ** 2 + dx ** 2 + dy ** 2)
        return gradT(c * dx, c * dy)

    def cmax(self, shape):
        if self.kind == 'quad':
            return self.beta * np.ones(shape)
        return (self.beta / self.k) * np.ones(shape)


def pwls(A, P, W, prior, U0, n_iter=150, momentum=True, track=False, A1=None,
         U_ref=None, nonneg=True, step=None, restart=False):
    # SQS with Nesterov momentum (Algorithm 1). Solves B problems at once:
    # P, W are (M, B), U0 is (N, N, B).
    N = U0.shape[0]; B = U0.shape[2]; npx = N * N
    if A1 is None:
        A1 = A @ np.ones(npx, dtype=np.float32)
    D = (A.T @ (W * A1[:, None]).astype(np.float32)).reshape(N, N, B)
    D = D + reg_majorizer(prior.cmax((N, N, B)))
    if step is not None:
        D = np.full_like(D, 1.0 / step)
    x = U0.copy(); z = x.copy(); t = 1.0
    hist = []

    def obj(u):
        r = (A @ u.reshape(npx, B).astype(np.float32)) - P
        return 0.5 * np.sum(W * r ** 2, axis=0) + prior.value(u)

    for it in range(n_iter):
        r = (A @ z.reshape(npx, B).astype(np.float32)) - P
        g = (A.T @ (W * r).astype(np.float32)).reshape(N, N, B) + prior.gradient(z)
        xn = z - g / D
        if nonneg:
            xn = np.maximum(xn, 0)
        if momentum and restart:
            bad = np.sum((z - xn) * (xn - x), axis=(0, 1)) > 0
            if np.any(bad):
                t = np.where(bad, 1.0, t) if np.ndim(t) else np.where(bad, 1.0, np.full(B, t))
        if momentum:
            tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            z = xn + ((t - 1) / tn) * (xn - x)
            t = tn
        else:
            z = xn
        x = xn
        if track:
            rec = {'obj': obj(x)}
            if U_ref is not None:
                rec['rmse'] = np.sqrt(np.mean((x - U_ref) ** 2, axis=(0, 1)))
            hist.append(rec)
    return x, hist, D


def perona_malik(u0, K, n_iter, dt=0.2):
    u = u0[..., None].copy()
    for _ in range(n_iter):
        dx, dy = grad(u)
        c = 1.0 / (1.0 + (dx ** 2 + dy ** 2) / K ** 2)
        u = u - dt * gradT(c * dx, c * dy)
    return u[..., 0]
