# figures of the paper, written to figs/
import json, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ctsim import MU_W, mu2hu

OUT = 'figs'; os.makedirs(OUT, exist_ok=True)
MM = 1 / 25.4; W = 174 * MM
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial', 'Helvetica'],
                     'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.linewidth': 0.6, 'lines.linewidth': 1.2, 'grid.linewidth': 0.4, 'axes.grid': True,
                     'grid.alpha': 0.3, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.framealpha': 0.9,
                     'legend.edgecolor': '0.8'})
OI = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73', 'red': '#D55E00',
      'purple': '#CC79A7', 'sky': '#56B4E9', 'black': '#000000', 'grey': '#7F7F7F'}
COL = [OI['blue'], OI['red'], OI['green'], OI['orange']]
LS = ['-', '--', '-.', ':']
MK = ['o', 's', '^', 'v']


def plabel(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, va='bottom', ha='left', fontsize=9, fontweight='bold')


def save(fig, name, vector):
    fig.savefig(f'{OUT}/{name}.pdf')
    if vector:
        fig.savefig(f'{OUT}/{name}.eps')
    fig.savefig(f'{OUT}/{name}.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)


def fig1_phantoms():
    fig, ax = plt.subplots(2, 3, figsize=(W, 0.60 * W))
    for c, (ph, lab) in enumerate([('head', 'Head'), ('thorax', 'Thorax'), ('lc', 'LC')]):
        x = np.load(f'data/gt_{ph}.npy')
        ax[0, c].imshow(mu2hu(x), cmap='gray', vmin=-160, vmax=240); ax[0, c].axis('off')
        ax[0, c].text(0.0, 1.02, f'({"abc"[c]}) {lab}', transform=ax[0, c].transAxes, fontsize=9, fontweight='bold')
        s = np.load(f'results/img_{ph}_5000.npz')['sig'] / MU_W * 1000
        s = np.where(x > MU_W * 0.01, s, np.nan)
        h = ax[1, c].imshow(s, cmap='magma'); ax[1, c].axis('off')
        cb = plt.colorbar(h, ax=ax[1, c], fraction=0.046, pad=0.02); cb.set_label(r'$\hat\sigma$ (HU)')
        ax[1, c].text(0.0, 1.02, f'({"def"[c]})', transform=ax[1, c].transAxes, fontsize=9, fontweight='bold')
    fig.tight_layout(pad=0.4); save(fig, 'Fig1', vector=False)


def fig2_noise():
    J = json.load(open('results/res_noise.json'))
    fig = plt.figure(figsize=(W, 0.72 * W))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], height_ratios=[0.8, 1], wspace=0.35, hspace=0.30,
                          left=0.08, right=0.92, bottom=0.08, top=0.95)
    axa, axb, cax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    axc, axd = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1:])
    Z = np.load('results/res_noise_thorax_5000.npz'); x = np.load('data/gt_thorax.npy'); body = x > MU_W * 0.01
    e, p = Z['emp'] / MU_W * 1000, Z['est'] / MU_W * 1000
    for a, im, l in [(axa, e, '(a)'), (axb, p, '(b)')]:
        h = a.imshow(np.where(body, im, np.nan)[30:226], cmap='magma', vmin=15, vmax=80); a.axis('off')
        a.text(0.0, 1.02, l, transform=a.transAxes, fontsize=9, fontweight='bold', va='bottom')
    cb = fig.colorbar(h, cax=cax); cb.set_label('Noise s.d. (HU)')
    idx = np.random.default_rng(0).choice(np.where(body.ravel())[0], 4000, replace=False)
    axc.scatter(e.ravel()[idx], p.ravel()[idx], s=1.2, alpha=0.35, color=OI['blue'], rasterized=True, lw=0)
    lim = [15, 85]
    axc.plot(lim, lim, 'k--', lw=0.8, label='Identity')
    axc.set_xlim(lim); axc.set_ylim(lim); axc.set_aspect('equal')
    axc.set_xticks([20, 40, 60, 80]); axc.set_yticks([20, 40, 60, 80])
    axc.set_xlabel('Monte Carlo s.d. (HU)'); axc.set_ylabel('Predicted s.d. (HU)')
    J5 = J['thorax_5000']
    axc.text(0.96, 0.05, f"$r$ = {J5['corr_est']:.3f}\nmedian error = {100 * J5['medrel_est']:.1f}%",
             transform=axc.transAxes, ha='right', va='bottom', fontsize=7,
             bbox=dict(boxstyle='round,pad=0.25', fc='w', ec='0.8'))
    plabel(axc, '(c)', x=-0.22)
    for i, (key, lab) in enumerate([('thorax_20000', r'Thorax, $I_0=2\times10^4$'), ('thorax_5000', r'Thorax, $I_0=5\times10^3$'),
                                    ('lc_20000', r'LC, $I_0=2\times10^4$'), ('lc_5000', r'LC, $I_0=5\times10^3$')]):
        r = np.array(J[key]['sino_ratio'])
        axd.semilogx(np.sqrt(r[:, 0] * r[:, 1]), r[:, 3], marker=MK[i], ls=LS[i], ms=4, color=COL[i], label=lab)
    axd.axhline(1, color='0.4', lw=0.7)
    axd.set_xlim(5, 3e4); axd.set_ylim(0.97, 1.30)
    axd.set_xlabel(r'Mean count $\bar y$'); axd.set_ylabel('Variance ratio (MC / delta method)')
    axd.legend(loc='upper right', fontsize=6.5)
    plabel(axd, '(d)', x=-0.14)
    save(fig, 'Fig2', vector=False)


def fig3_conv():
    C = json.load(open('results/res_conv.json')); K = json.load(open('results/res_consistency.json')); E = json.load(open('results/res_euler.json'))
    fig, ax = plt.subplots(1, 3, figsize=(W, 0.36 * W))
    names = [('GD (1/L)', 'Gradient descent, step $1/L$'), ('SQS', 'SQS'), ('SQS-Nesterov', 'SQS-Nesterov')]
    for i, (nm, lab) in enumerate(names):
        g = np.maximum(np.array(C[nm]['gap']), 1e-12) / abs(C['Fstar'])
        ax[0].loglog(np.arange(1, len(g) + 1), g, ls=LS[i], color=COL[i], label=lab)
    n = np.arange(1, 401)
    ax[0].loglog(n, 2 * C['D_norm_u0'] / (n + 1) ** 2 / abs(C['Fstar']), color=COL[2], lw=0.9, ls=':',
                 label='Bound for SQS-Nesterov\n(Theorem 2)')
    ax[0].set_xlabel('Iteration $n$'); ax[0].set_ylabel(r'$(F(u^{(n)})-F^*)/|F^*|$')
    ax[0].legend(loc='lower left', fontsize=6)
    plabel(ax[0], '(a)', x=-0.30)
    for i, fac in enumerate(['1.0', '1.9', '2.05']):
        o = np.array(E[fac]); ax[1].semilogy(np.arange(1, len(o) + 1), o, ls=LS[i], color=COL[i], label=rf'$\tau={fac}/L$')
    ax[1].set_xlabel('Iteration $n$'); ax[1].set_ylabel(r'$F(u^n)$'); ax[1].legend(loc='upper left', fontsize=6.5)
    ax[1].set_ylim(1e5, 1e14)
    ins = ax[1].inset_axes([0.50, 0.10, 0.46, 0.30])
    for i, fac in enumerate(['1.0', '1.9']):
        o = np.array(E[fac]); ins.plot(np.arange(1, len(o) + 1), o / 1e5, ls=LS[i], color=COL[i])
    ins.text(0.95, 0.88, r'linear scale, $\times10^5$', transform=ins.transAxes, ha='right', va='top', fontsize=5.5)
    ins.tick_params(labelsize=5.5); ins.grid(alpha=0.3)
    plabel(ax[1], '(b)', x=-0.30)
    e = np.array(K['consistency']['errors'])
    ax[2].loglog(e[:, 0], e[:, 1], 'o-', ms=3.5, color=COL[0], label='Max error')
    ax[2].loglog(e[:, 0], e[:, 2], 's--', ms=3.5, color=COL[1], label='RMS error')
    ax[2].loglog(e[:, 0], e[-1, 2] * e[-1, 0] / e[:, 0], color='0.4', lw=0.9, ls=':', label='Slope $-1$')
    ax[2].axvspan(25, 80, color='0.9', zorder=0)
    ax[2].text(45, 0.17, 'under-\nresolved', ha='center', va='bottom', fontsize=6, color='0.3')
    ax[2].set_xticks([32, 64, 128, 256, 512]); ax[2].set_xticklabels(['32', '64', '128', '256', '512']); ax[2].minorticks_off()
    ax[2].set_xlim(25, 650); ax[2].set_ylim(0.15, 100)
    ax[2].set_xlabel('Grid size $N$'); ax[2].set_ylabel('Consistency error'); ax[2].legend(loc='upper right', fontsize=6.5)
    plabel(ax[2], '(c)', x=-0.30)
    fig.tight_layout(pad=0.4, w_pad=1.2); save(fig, 'Fig3', vector=True)


def fig4_images(tag='thorax_5000', zoom=(slice(40, 140), slice(60, 196))):
    Z = np.load(f'results/img_{tag}.npz')
    names = [('gt', 'Ground truth'), ('FBP', 'FBP'), ('PWLS_U', 'PWLS-U'), ('PWLS_CB', 'PWLS-CB'), ('PWLS_NA', 'PWLS-NA')]
    gt = mu2hu(Z['gt'])
    fig = plt.figure(figsize=(W, 0.56 * W))
    hr = [1.0, (zoom[0].stop - zoom[0].start) / (zoom[1].stop - zoom[1].start), 1.0]
    gs = fig.add_gridspec(3, 6, width_ratios=[1, 1, 1, 1, 1, 0.06], height_ratios=hr,
                          left=0.045, right=0.89, top=0.94, bottom=0.01, wspace=0.04, hspace=0.05)
    for c, (k, t) in enumerate(names):
        im = mu2hu(Z[k])
        a0 = fig.add_subplot(gs[0, c]); h0 = a0.imshow(im, cmap='gray', vmin=-160, vmax=240)
        a0.set_title(t, fontsize=8, pad=3)
        r0, c0 = zoom[0].start, zoom[1].start
        a0.add_patch(plt.Rectangle((c0, r0), zoom[1].stop - c0, zoom[0].stop - r0, fill=False, ec=OI['orange'], lw=0.7))
        a1 = fig.add_subplot(gs[1, c]); h1 = a1.imshow(im[zoom], cmap='gray', vmin=-60, vmax=140)
        a2 = fig.add_subplot(gs[2, c])
        if k == 'gt':
            a2.axis('off')
        else:
            h2 = a2.imshow(im - gt, cmap='gray', vmin=-100, vmax=100)
        for a in (a0, a1, a2):
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values(): sp.set_visible(False)
        if c == 0:
            for a, l in [(a0, '(a)'), (a1, '(b)'), (a2, '(c)')]:
                a.text(-0.06, 0.5, l, transform=a.transAxes, ha='right', va='center', fontsize=9, fontweight='bold')
    for r, h, lab in [(0, h0, 'HU'), (1, h1, 'HU'), (2, h2, 'Error (HU)')]:
        cb = fig.colorbar(h, cax=fig.add_subplot(gs[r, 5])); cb.set_label(lab, fontsize=7); cb.ax.tick_params(labelsize=6)
    save(fig, 'Fig4', vector=False)


def fig5_design():
    G = json.load(open('results/res_design.json')); NR = json.load(open('results/res_nr.json'))
    fig, ax = plt.subplots(1, 3, figsize=(W, 0.36 * W))
    maps = ['fbp_g0', 'fbp_g1', 'fbp_g2', 'fbp_g3', 'cert_e0.5', 'cert_e1.0', 'cert_e1.5']
    lab = [r'$\hat\sigma^{0}$', r'$\hat\sigma^{-1}$', r'$\hat\sigma^{-2}$', r'$\hat\sigma^{-3}$',
           r'$\kappa^{1}$', r'$\kappa^{2}$', r'$\kappa^{3}$']
    xs = np.arange(len(maps)); wd = 0.27
    for i, (ph, phl, hatch) in enumerate([('head', 'Head', ''), ('thorax', 'Thorax', '///'), ('lc', 'LC', '...')]):
        best = [min(G[f'{ph}_{m}'], key=lambda r: r['rmse']) for m in maps]
        ax[0].bar(xs + (i - 1) * wd, [b['rmse'] - best[0]['rmse'] for b in best], wd, label=phl,
                  color=COL[i], edgecolor='k', linewidth=0.4, hatch=hatch, alpha=0.85)
    ax[0].set_ylim(-8.5, 6.8); ax[0].axhline(0, color='k', lw=0.6); ax[0].set_xticks(xs); ax[0].set_xticklabels(lab)
    ax[0].axvline(3.5, color='0.5', lw=0.6, ls='--')
    ax[0].text(1.5, 0.97, 'FBP-noise map', transform=ax[0].get_xaxis_transform(), ha='center', va='top', fontsize=6.5)
    ax[0].text(5.0, 0.97, 'Certainty map', transform=ax[0].get_xaxis_transform(), ha='center', va='top', fontsize=6.5)
    ax[0].set_ylabel('RMSE change vs. uniform (HU)'); ax[0].legend(loc='lower left', fontsize=6.5)
    plabel(ax[0], '(a)', x=-0.28)
    for i, m in enumerate(['PWLS-U', 'PWLS-CB', 'PWLS-NA']):
        r = NR[m]; nm = m
        for a, key in [(ax[1], 'noise_near'), (ax[2], 'noise_far')]:
            a.plot([q['esf_10_90_mm'] for q in r], [q[key] for q in r], color=COL[i], ls=LS[i], marker=MK[i], ms=3.5, label=nm)
    for a, l in [(ax[1], '(b)'), (ax[2], '(c)')]:
        a.set_xlabel('ESF 10-90% width (mm)'); a.set_ylabel('Noise s.d. (HU)'); a.legend(loc='upper right', fontsize=6.5)
        plabel(a, l, x=-0.24)
    fig.tight_layout(pad=0.4, w_pad=1.0); save(fig, 'Fig5', vector=True)


if __name__ == '__main__':
    for f in [fig1_phantoms, fig2_noise, fig3_conv, fig4_images, fig5_design]:
        f(); print('ok', f.__name__)
