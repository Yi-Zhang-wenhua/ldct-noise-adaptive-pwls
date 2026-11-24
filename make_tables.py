# LaTeX tables of the paper, written to tables/
import json, numpy as np
from scipy.stats import wilcoxon
M = ['FBP', 'FBP+PM', 'FBP+TV', 'FBP+BM3D', 'PWLS-Q', 'PWLS-TV', 'PWLS-U', 'PWLS-CB', 'PWLS-NA']
LAB = {m: m for m in M}
PH = [('head', 'head'), ('thorax', 'thorax'), ('lc', 'LC')]; DO = [(20000, 'D1'), (10000, 'D2'), (5000, 'D3')]
K = ['rmse', 'rmse_smooth', 'rmse_edge', 'ssim']
R = {(p, d): json.load(open(f'results/res_main_{p}_{d}.json'))['test'] for p, _ in PH for d, _ in DO}
tex = ''
for p, pl in PH:
    rows = []
    for d, dl in DO:
        T = R[(p, d)]
        v = {m: {k: np.array([r[k] for r in T[m]]) for k in K} for m in M}
        best = {k: (min if k != 'ssim' else max)(M, key=lambda m: v[m][k].mean()) for k in K}
        for i, m in enumerate(M):
            cells = []
            for k in K:
                x = v[m][k]
                c = f'{x.mean():.3f} ({x.std(ddof=1):.3f})' if k == 'ssim' else f'{x.mean():.1f} ({x.std(ddof=1):.1f})'
                cells.append(r'\textbf{' + c + '}' if m == best[k] else c)
            lead = r'\multirow{9}{*}{' + dl + '}' if i == 0 else ''
            rows.append(f'{lead} & {LAB[m]} & ' + ' & '.join(cells) + r' \\')
        rows.append(r'\midrule')
    rows[-1] = r'\bottomrule'
    tex += (r'\begin{table}[tbp]\centering' + '\n' + r'\caption{Test results for the ' + pl + r' phantom: mean (standard deviation) over six noise realisations. RMSE values are in HU. Bold marks the best mean at each dose}\label{tab:main_' + p + '}\n'
            r'\begin{tabular}{llcccc}\toprule' + '\n' + r'Dose & Method & RMSE & RMSE smooth & RMSE edge & SSIM\\\midrule' + '\n' + '\n'.join(rows) + '\n' + r'\end{tabular}' + '\n' + r'\end{table}' + '\n\n')
open('tables/tab_main.tex', 'w').write(tex)

def stat(m, k):
    a = np.concatenate([[r[k] for r in R[c]['PWLS-NA']] for c in R]); b = np.concatenate([[r[k] for r in R[c][m]] for c in R])
    wins = int(np.sum((a < b) if k != 'ssim' else (a > b))); p = wilcoxon(a, b).pvalue; md = np.median(a - b)
    ps = r'$<10^{-4}$' if p < 1e-4 else f'{p:.3f}'
    return '$' + (f'{md:+.4f}' if k == 'ssim' else f'{md:+.2f}') + '$', f'{wins}/{len(a)}', ps
blocks = []
for title, ks in [('(a) RMSE over the object and in smooth regions', [('rmse', 'RMSE'), ('rmse_smooth', 'RMSE smooth')]),
                  ('(b) RMSE in the edge band and SSIM', [('rmse_edge', 'RMSE edge'), ('ssim', 'SSIM')])]:
    rows = [f'{LAB[m]} & ' + ' & '.join(' & '.join(stat(m, k)) for k, _ in ks) + r' \\' for m in M if m != 'PWLS-NA']
    blocks.append(r'\multicolumn{7}{l}{' + title + r'}\\\midrule' + '\n'
                  + r' & \multicolumn{3}{c}{' + ks[0][1] + r'} & \multicolumn{3}{c}{' + ks[1][1] + r'}\\' + '\n'
                  + r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}' + '\n'
                  + r'Method & $\Delta$ & Wins & $p$ & $\Delta$ & Wins & $p$\\\midrule' + '\n' + '\n'.join(rows))
tex = ('\\begin{table}[tbp]\\centering\n\\caption{Paired comparison of PWLS-NA with each other method over all nine cases and six realisations (54 pairs). $\\Delta$ is the median of (PWLS-NA $-$ method); for RMSE a negative value favours PWLS-NA, for SSIM a positive value does. ``Wins\'\' counts the pairs in which PWLS-NA is better; $p$ is from the two-sided Wilcoxon signed-rank test}\\label{tab:stats}\n'
       r'\begin{tabular}{lcccccc}\toprule' + '\n' + ('\n' + r'\midrule' + '\n').join(blocks) + '\n' + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n' + r'\end{table}' + '\n')
open('tables/tab_stats.tex', 'w').write(tex)
print('ok')
