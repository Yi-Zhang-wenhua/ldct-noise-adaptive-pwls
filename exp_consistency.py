# consistency order of the discrete diffusion operator (manufactured solution)
import json
import numpy as np
import sympy as s
from recon import grad, gradT

x, y = s.symbols('x y'); k = s.Rational(1, 2)
U = s.sin(2 * s.pi * x) * s.cos(2 * s.pi * y) * s.exp(-((x - s.Rational(1, 2)) ** 2 + (y - s.Rational(1, 2)) ** 2) / s.Rational(5, 100))
Bt = 1 + s.Rational(1, 2) * s.sin(s.pi * x) * s.sin(s.pi * y)
ux, uy = s.diff(U, x), s.diff(U, y)
r = s.sqrt(k ** 2 + ux ** 2 + uy ** 2)
Lf = s.lambdify((x, y), s.diff(Bt * ux / r, x) + s.diff(Bt * uy / r, y), 'numpy')
Bf = s.lambdify((x, y), Bt, 'numpy')
Uf = s.lambdify((x, y), U, 'numpy')

errs = []
for n in [32, 64, 128, 256, 512]:
    h = 1.0 / n
    c = (np.arange(n) + 0.5) * h
    Y, X = np.meshgrid(c, c, indexing='ij')
    u = Uf(X, Y)
    dx, dy = grad(u[..., None]); dx = dx[..., 0] / h; dy = dy[..., 0] / h
    cc = Bf(X, Y) / np.sqrt(0.25 + dx ** 2 + dy ** 2)
    Ld = -gradT((cc * dx)[..., None], (cc * dy)[..., None])[..., 0] / h
    m = (X > 0.2) & (X < 0.8) & (Y > 0.2) & (Y < 0.8)
    d = (Ld - Lf(X, Y))[m]
    errs.append((n, float(np.max(np.abs(d))), float(np.sqrt(np.mean(d ** 2)))))
orders = [(errs[i][0], float(np.log2(errs[i - 1][1] / errs[i][1])), float(np.log2(errs[i - 1][2] / errs[i][2])))
          for i in range(1, len(errs))]
print(errs); print(orders)
json.dump({'consistency': {'errors': errs, 'orders': orders}}, open('results/res_consistency.json', 'w'), indent=1)
