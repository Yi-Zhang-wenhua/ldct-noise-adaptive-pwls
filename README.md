# Noise-adaptive penalty weighting for low-dose CT

Code for the paper "Noise-adaptive penalty weighting for edge-preserving diffusion in low-dose CT reconstruction: analysis, numerical verification and a controlled simulation study" (Y. Zhang, B. Pang).

## Files

| File | Contents |
|---|---|
| `ctsim.py` | phantoms, analytic projections, projectors, noise model, FBP, FBP noise map |
| `recon.py` | PWLS solver (SQS with Nesterov momentum), penalties, Perona-Malik |
| `prepare.py` | builds the projectors, ground truth and sinograms in `data/` |
| `exp_noise.py` | noise formula vs. Monte Carlo (Table 1, Fig. 2) |
| `exp_consistency.py` | consistency order of the discrete operator (Fig. 3c) |
| `exp_main.py` | main comparison of nine methods (Tables 2-5, Fig. 4) |
| `exp_extra.py` | convergence, stability, noise-resolution, design and hybrid studies (Figs. 3, 5) |
| `make_tables.py`, `make_figures.py` | tables and figures from `results/` |
| `results/` | raw results used in the paper |

## Run

Python 3.13 was used.

```
pip install -r requirements.txt
python prepare.py
python exp_noise.py
python exp_consistency.py
python exp_main.py            # or split over workers: python exp_main.py 0 2 and python exp_main.py 1 2
python exp_extra.py conv
python exp_extra.py euler
python exp_extra.py nr
python exp_extra.py design
python exp_extra.py hybrid
python make_tables.py
python make_figures.py
```

`exp_main.py` skips cases that already have a result file in `results/`, so delete those files to run it again. All random seeds are fixed. On a 2-core machine the main study takes about 3 hours.

## Licence

MIT
