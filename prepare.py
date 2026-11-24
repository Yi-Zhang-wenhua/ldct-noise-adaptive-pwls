# builds the projectors, ground-truth images and analytic sinograms in data/
import os
import numpy as np
import scipy.sparse as sp
from ctsim import (Geometry, system_matrix, system_matrix_sf, raster, analytic_sinogram,
                   phantom_head, phantom_thorax, phantom_lowcontrast)

os.makedirs('data', exist_ok=True)
g = Geometry()
sp.save_npz('data/A.npz', system_matrix(g))
sp.save_npz('data/Asf.npz', system_matrix_sf(g))
for name, ph in [('head', phantom_head()), ('thorax', phantom_thorax()), ('lc', phantom_lowcontrast())]:
    np.save(f'data/gt_{name}.npy', raster(ph, g))
    np.save(f'data/sino_{name}.npy', analytic_sinogram(ph, g))
    print(name, 'done')
