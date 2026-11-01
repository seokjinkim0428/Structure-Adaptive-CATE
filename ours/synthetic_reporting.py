"""Dimension-based tables and scientific plots shared by the demo and manuscript."""
from pathlib import Path

import numpy as np
import pandas as pd

from ours.synthetic_experiments import METHODS, normalize_method_names

COLORS = ('#0072B2', '#80858F', '#009E73', '#E69F00', '#CC79A7')
MARKERS = ('o', 's', '^', 'D', 'v')
SETTINGS = ((10, 10), (10, 4), (20, 20), (20, 4))


def comparison_rows(summary, dimension):
    """All comparisons in one sequence per dimension, with explicit source keys.

    Identical noise/overlap reference rows and the default representation row
    appear once. Single-split rows change Ours only and reuse the unchanged
    baselines from the same datasets and kernels, as indicated in the caption.
    """
    summary = normalize_method_names(summary)
    if dimension not in ('univariate', 'multivariate'):
        raise ValueError('dimension must be univariate or multivariate')
    data = summary[summary.d.eq(1) if dimension == 'univariate' else summary.d.gt(1)]
    indexed = summary.set_index(['condition_id', 'method'])
    assert indexed.index.is_unique
    conditions = data.drop_duplicates('condition_id')
    rows = []

    def append(cid, label, ours_source=None, baseline_source=None):
        condition = conditions[conditions.condition_id.eq(cid)].iloc[0]
        sources = {}
        for method in METHODS:
            sources[method] = (baseline_source or cid, method)
        if ours_source is not None:
            sources['Ours'] = (cid, ours_source)
        # Quick runs may omit some comparison groups; no missing scores are invented.
        if not all(source in indexed.index for source in sources.values()):
            raise ValueError(f'Missing five-method result for {cid}')
        counts = [int(indexed.loc[source, 'n_eff']) for source in sources.values()]
        assert len(set(counts)) == 1, (cid, counts)
        rows.append(dict(condition_id=cid, d=int(condition.d), p=int(condition.p),
                         n=int(condition.n_samples), replications=counts[0],
                         label=label, sources=sources))

    settings = [(1, 1)] if dimension == 'univariate' else [s for s in SETTINGS if ((data.d==s[0]) & (data.p==s[1])).any()]
    for d, p in settings:
        case = conditions[conditions.d.eq(d) & conditions.p.eq(p)]
        for row in case[case.study.eq('original')].sort_values('n_samples').itertuples():
            append(row.condition_id, 'Sample size')
        for row in case[case.study.eq('noise')].sort_values('noise_std').itertuples():
            append(row.condition_id, rf'$\sigma={row.noise_std:g}$')
        overlap = case[case.study.isin(['kappa','overlap_extension'])].sort_values('kappa')
        for row in overlap.itertuples():
            if row.kappa == .1 and case[case.study.eq('noise') & case.noise_std.eq(1)].shape[0]:
                continue
            append(row.condition_id, rf'$\kappa={row.kappa:g}$')
        if d == 1:
            for cid,label in [('nuisance__h1_h2',r'$H^1/H^2$'),('nuisance__h2_h3',r'$H^2/H^3$')]:
                if cid in set(case.condition_id): append(cid,label)
        if (d,p) == (20,4):
            if 'representation__correct' in set(case.condition_id) and not (
                case.study.eq('noise') & case.noise_std.eq(1)
            ).any():
                append('representation__correct', 'Default library')
            for cid,label in [('representation__correct_plus_wrong',r'Add $x_{5:8}$'),('representation__wrong_only',r'$x_{5:8}$ only')]:
                if cid in set(case.condition_id): append(cid,label)
        cid=f'crossfit__d{d}_p{p}'
        if cid in set(case.condition_id):
            append(cid,'Single split',ours_source='Ours single split',baseline_source=f'noise__d{d}_p{p}_sigma1')
    return rows


def comparison_table(summary, dimension):
    summary = normalize_method_names(summary)
    indexed = summary.set_index(['condition_id','method'])
    records = []
    for row in comparison_rows(summary, dimension):
        entry = dict(d=row['d'],p=row['p'],configuration=row['label'],n=row['n'],replications=row['replications'])
        for method,source in row['sources'].items():
            value=indexed.loc[source]
            entry[method]=f'{value.mise_mean:.4f} ({value.mise_se:.4f})'
        records.append(entry)
    return pd.DataFrame(records)


def _style():
    import matplotlib.pyplot as plt
    return plt.rc_context({'font.family':'DejaVu Sans','font.size':9.5,
                           'axes.titlesize':10,'axes.labelsize':9,
                           'xtick.labelsize':8,'ytick.labelsize':8,
                           'axes.spines.top':False,'axes.spines.right':False,
                           'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})


def _panel(ax, data, x, title, xlabel, categorical=None):
    if data.empty:
        ax.text(.5,.5,'Not included in this profile',ha='center',va='center',transform=ax.transAxes)
        ax.set_axis_off()
        return
    for method,color,marker in zip(METHODS,COLORS,MARKERS):
        part=data[data.method.eq(method)].sort_values(x)
        if part.empty: continue
        xx=part[x].to_numpy() if categorical is None else np.array([categorical.index(v) for v in part[x]])
        ax.errorbar(xx,part.mise_mean,yerr=part.mise_se,color=color,marker=marker,
                    linewidth=1.7 if method=='Ours' else 1.05,markersize=3.6,
                    elinewidth=.8,capsize=2,label=method,zorder=4 if method=='Ours' else 2)
    ax.set_yscale('log')
    if categorical is None:
        vals=sorted(data[x].unique())
        if x in ('noise_std','kappa'): ax.set_xscale('log')
        ax.set_xticks(vals,labels=[f'{v:g}' for v in vals])
    else:
        ax.set_xticks(range(len(categorical)), labels=['H¹ / H²','H² / H³'])
    ax.minorticks_off()
    ax.set_title(title,loc='left',fontweight='bold',pad=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Mean MISE')
    ax.grid(axis='y',alpha=.17,linewidth=.6)
    ax.margins(x=.09)


def _legend(fig):
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color=c,marker=m,linewidth=1.7 if name=='Ours' else 1.05,
                    markersize=4,label=name) for name,c,m in zip(METHODS,COLORS,MARKERS)]
    fig.legend(handles=handles,loc='outside lower center',ncol=5,frameon=False,
               columnspacing=1.35,handlelength=1.7,fontsize=9)


def plot_univariate_comparison(summary):
    summary = normalize_method_names(summary)
    import matplotlib.pyplot as plt
    data=summary[summary.d.eq(1)]
    with _style():
        fig,axes=plt.subplots(2,2,figsize=(7.2,5.1),layout='constrained')
        _panel(axes[0,0],data[data.study.eq('original')],'n_samples',
               '(a) Sample size','Training sample size n')
        _panel(axes[0,1],data[data.study.eq('noise')],'noise_std',
               '(b) Noise','Noise standard deviation σ')
        _panel(axes[1,0],data[data.study.isin(['kappa','overlap_extension'])],'kappa',
               '(c) Overlap','Propensity clipping κ')
        _panel(axes[1,1],data[data.study.eq('nuisance')],'condition_id',
               '(d) Kernel misspecification','Nuisance / target kernel',
               categorical=['nuisance__h1_h2','nuisance__h2_h3'])
        if len(data) and data.profile.iloc[0]=='quick':
            fig.suptitle('Quick run',fontsize=10)
            for ax in axes.flat: ax.set_title(ax.get_title().split(' · ')[0],loc='left')
        _legend(fig)
    return fig


def plot_multivariate_comparison(summary):
    summary = normalize_method_names(summary)
    import matplotlib.pyplot as plt
    data=summary[summary.d.gt(1)]
    with _style():
        fig,axes=plt.subplots(5,2,figsize=(7.2,9.0),layout='constrained')
        for ax,(d,p),letter in zip(axes[0],[(10,10),(10,4)],['a','b']):
            part=data[data.study.eq('original') & data.d.eq(d) & data.p.eq(p)]
            _panel(ax,part,'n_samples',f'({letter}) d = 10, '+('dense' if p==d else 'sparse')+' · n',
                   'Training sample size n')
        for i,(d,p) in enumerate(SETTINGS,1):
            case=data[data.d.eq(d) & data.p.eq(p)]
            name=f'd = {d}, '+('dense' if p==d else 'sparse (p = 4)')
            _panel(axes[i,0],case[case.study.eq('noise')],'noise_std',
                   f'({chr(97+2*i)}) {name} · noise','Noise standard deviation σ')
            _panel(axes[i,1],case[case.study.isin(['kappa','overlap_extension'])],'kappa',
                   f'({chr(98+2*i)}) {name} · overlap','Propensity clipping κ')
        if len(data) and data.profile.iloc[0]=='quick':
            fig.suptitle('Quick run',fontsize=10)
        _legend(fig)
    return fig


def save_comparison_figures(summary, output_dir):
    import matplotlib.pyplot as plt
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    paths={}
    for name,plotter in [('univariate',plot_univariate_comparison),('multivariate',plot_multivariate_comparison)]:
        fig=plotter(summary)
        for suffix in ['pdf','png','svg']:
            path=output/f'synthetic_{name}.{suffix}'
            fig.savefig(path,dpi=200,bbox_inches='tight')
            paths[f'{name}_{suffix}']=path
        plt.close(fig)
    return paths


def latex_comparison_table(summary, dimension):
    """One complete table per dimension; every numeric cell has a source key."""
    summary = normalize_method_names(summary)
    indexed=summary.set_index(['condition_id','method'])
    multi=dimension=='multivariate'
    rows=comparison_rows(summary,dimension)
    common=(r'Entries are mean MISE (SE), with the smallest mean in bold. '
            r'Unless varied, $\sigma=1$ and $\kappa=0.1$; the $\sigma=1$ row also supplies the $\kappa=0.1$ reference. '
            r'Single-split rows change Ours only and repeat the unchanged matched baselines. ')
    caption=('Multivariate synthetic results. '+common+
             r'$p=d$ is dense and $p=4$ is sparse. Sample-size rows use fixed kernel lengths; other rows use median-calibrated lengths. '
             r'The $\sigma=1$ row at $(d,p)=(20,4)$ also gives the default representation library.'
             if multi else 'Univariate synthetic results. '+common+
             r'$H^a/H^b$ denotes the jointly varied nuisance/target Sobolev orders; $H^2$ misspecifies the kinked nuisance.')
    lines=[r'\begin{table}[htbp]',r'\centering',r'\caption{'+caption+'}',
           r'\label{tab:'+dimension+'-all}']
    aliases=['table; multivariate n=1000','table; multivariate n=2000'] if multi else ['table:univariate']
    lines += [r'\label{'+label+'}' for label in aliases]
    lines += [r'\fontsize{8.2}{10}\selectfont' if multi else r'\fontsize{8.5}{10}\selectfont',
              r'\setlength{\tabcolsep}{1.8pt}' if multi else r'\setlength{\tabcolsep}{2pt}',
              r'\begin{tabular}{@{}'+('l' if multi else '')+'lrrccccc@{}}',r'\toprule',
              ('$(d,p)$ & ' if multi else '')+r'Configuration & $n$ & Reps. & Ours & Plug-in & DR & Oracle DR & R \\',r'\midrule']
    previous=None
    for row in rows:
        setting=(row['d'],row['p'])
        if multi and previous is not None and setting!=previous:lines.append(r'\addlinespace[3pt]')
        previous=setting
        cells=([f"$({row['d']},{row['p']})$"] if multi else [])+[row['label'],str(row['n']),str(row['replications'])]
        best=min(METHODS,key=lambda method:indexed.loc[row['sources'][method],'mise_mean'])
        for method in METHODS:
            value=indexed.loc[row['sources'][method]]
            number=f'{value.mise_mean:.4f}'
            if method==best:number=r'\textbf{'+number+'}'
            cells.append(number+r'\,'+f'({value.mise_se:.4f})')
        lines.append(' & '.join(cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
    return '\n'.join(lines)


def latex_synthetic_section(summary, n_evaluations, snapshot_name):
    """Table-based synthetic comparisons for the manuscript."""
    summary = normalize_method_names(summary)
    indexed = summary.set_index(['condition_id', 'method'])
    text = r"""\section{Numerical Experiments}\label{section; experiments}
% Full-run snapshot: manuscript_update/SNAPSHOT; EVALUATIONS scores.
We compare Ours with plug-in KRR, DR \citep{kennedy2023towards}, oracle-propensity DR, and R \citep{nie2021quasi}, using the same datasets and test covariates for all methods. HOIF methods serve as theoretical benchmarks.

The raw ridge grid is $\mathcal A=\{1,2,4,\ldots,512\}$ in $(K+\alpha I)^{-1}y$. Ours fixes nuisance ridge at $0.01$, corresponding to normalized regularization $0.01/n_{\rm stage}$, and selects the target kernel and ridge by Algorithm~\ref{algorithm; model selection}. We use untruncated predictions and average the three cyclic fits. Plug-in uses three-fold CV and refits each arm. DR, oracle DR, and R use three-fold CV within two swapped fits and average their predictions. R minimizes the residualized loss $(Y-\hat m(X)-(A-\hat\pi(X))h(X))^2$, with $m(x)=\mathbb E[Y\mid X=x]$. Estimated propensities are clipped to $[0.01,0.99]$; oracle DR uses the true propensity.

We generate $A\mid X\sim\operatorname{Bernoulli}(\pi_\kappa(X))$ and $Y=f_0^\star(X)+Ah^\star(X)+\varepsilon$, where $\pi_\kappa(x)=\operatorname{Clip}(\sin(5\|x\|_2),\kappa,1-\kappa)$ and $\varepsilon\sim\mathcal N(0,\sigma^2)$ is independent. Defaults are $\sigma=1$ and $\kappa=0.1$. Tables report mean MISE (Monte Carlo SE) on 3,000 independent uniform test points (seed $10099$), excluded from tuning. Sample-size comparisons use 100 replications; sweeps and ablations use 30, with seed $100+10r$ for replication $r$.

\subsection{Univariate Synthetic Data}
We draw $X\sim\operatorname{Unif}[0,1]$ and set
\[
f_0^\star(x)=5\bigl(|x-0.4|+|x-0.8|\bigr),\qquad h^\star(x)=x^2.
\]
The default nuisance/target kernels are unanchored Sobolev kernels of orders $(H^1,H^2)$. We vary $n\in\{500,1000\}$ and, at $n=500$, noise $\sigma\in\{0.25,0.5,1,2,4\}$ and overlap $\kappa\in\{0.01,0.02,0.05,0.1,0.2,0.3,0.5\}$. The $(H^2,H^3)$ ablation at $n=1000$ misspecifies the kinked nuisance while retaining a smooth target; it changes both kernels.

Table~\ref{tab:univariate-all} shows that Ours is competitive with DR, oracle DR, and R, with rankings depending on the setting. Ours has the lowest means at $\sigma=2,4$. Under the misspecified kernel pair, its MISE changes only from $0.0459$ to $0.0493$, compared with $0.0790$ to $0.3063$ for plug-in and $0.0440$ to $6.0731$ for DR. Cyclic averaging improves Ours from $0.1096$ to $0.0890$ at the default 30-replication setting.

""".replace('SNAPSHOT', snapshot_name).replace('EVALUATIONS', f'{n_evaluations:,}')
    text += latex_comparison_table(summary, 'univariate') + '\\clearpage\n'
    text += r"""
\subsection{Multivariate Synthetic Data}
We draw $X\sim\operatorname{Unif}([-1,1]^d)$ and set
\[
f_0^\star(x)=\frac{2}{d}\sum_{j=1}^{d}\sin(x_j),\qquad
h^\star(x)=\frac{0.3}{p}\sum_{j=1}^{p}x_j^2.
\]
We consider dense ($p=d$) and sparse ($p=4$) targets with $d=10,20$. The nuisance uses a Mat\'ern kernel with $\nu=1.5$. Ours selects among Mat\'ern ($\nu=1.5,2.5$) and RBF kernels on all coordinates and on $x_{1:4}$. DR, oracle DR, and R use Mat\'ern target kernels with $\nu=2.5$ and receive the true support in sparse settings. Sample-size comparisons use fixed kernel lengths; sweeps use lengths calibrated to correlation $0.5$ at the median distance on 2,000 independent covariates (seed $42$). Sweeps use $n=1000$, $\sigma\in\{0.5,1,2\}$, and $\kappa\in\{0.01,0.05,0.1,0.2,0.5\}$.

"""
    original = summary[summary.study.eq('original') & summary.d.gt(1)].condition_id.unique()
    count = sum(min(METHODS, key=lambda m: indexed.loc[(cid, m), 'mise_mean']) == 'Ours'
                for cid in original)
    text += (f'The advantage of Ours is clearer in Table~\\ref{{tab:multivariate-all}}: '
             f'it has the lowest mean in {count} of the {len(original)} sample-size comparisons '
             'and all 12 noise conditions. ')
    text += r"""At $\sigma=1$, its error is approximately $77$--$83\%$ below plug-in. R or DR leads in some moderate-overlap settings. Oracle DR need not dominate estimated-propensity DR in finite samples, since exact propensities can increase weighting variance.

For $(d,p)=(20,4)$, adding irrelevant $x_{5:8}$ candidates barely changes Ours ($0.0092$ to $0.0094$). Using only these incorrect coordinates gives $0.0065$, a finite-sample result rather than evidence of support recovery. Baselines keep their true-support kernels. Cyclic averaging reduces Ours's error by $33$--$53\%$ relative to a single split across the four multivariate settings.

"""
    text += latex_comparison_table(summary, 'multivariate')
    return text + '\\clearpage\n\n'
