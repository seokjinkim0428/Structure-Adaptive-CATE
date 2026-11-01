"""Evaluation-only RealCause noise correction using the scored realizations.

No estimator is fitted here. Saved MSEs are corrected by the across-realization
variance of sampled effects at each fixed test covariate. Joint delete-one
jackknife estimates account for reusing realizations in both quantities.
"""
from pathlib import Path
import hashlib
import io
import json

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.model_selection import train_test_split

from ours.realcause import map_df_to_XAYtau, read_sample_bytes, DEFAULT_DATA_DIR

METHOD_COLUMNS = {"Ours": "mse_ours", "Plug-in": "mse_plugin",
                  "DR": "mse_dr", "R": "mse_r"}


def noise_adjusted_statistics(mse, effects, outcome_ranges, method_names):
    """Correct an R-by-method score matrix using R-by-test-row raw effects.

    Realizations must be independent draws from one fixed generator, with the
    same test covariates and training/test row split. Outcome ranges come only
    from training rows. All methods share those ranges and scored test draws.
    The correction estimates CATE risk in expectation; finite-sample estimates
    can be negative and are never clipped. At least three realizations are
    required because every jackknife subset must support an unbiased variance.
    """
    mse = np.asarray(mse, dtype=float)
    effects = np.asarray(effects, dtype=float)
    ranges = np.asarray(outcome_ranges, dtype=float)
    if mse.ndim != 2 or effects.ndim != 2 or ranges.ndim != 1:
        raise ValueError("Expected score/effect matrices and a range vector.")
    n = len(mse)
    if n < 3 or len(effects) != n or len(ranges) != n or effects.shape[1] == 0:
        raise ValueError("Need at least three matched realizations and nonempty test rows.")
    if len(method_names) != mse.shape[1] or len(set(method_names)) != len(method_names):
        raise ValueError("One unique name is required per score column.")
    if not all(np.isfinite(a).all() for a in (mse, effects, ranges)) or (ranges <= 0).any():
        raise ValueError("Scores/effects must be finite and outcome ranges positive.")
    weights = ranges ** -2
    centered = effects - effects.mean(axis=0)
    row_squares = np.mean(centered ** 2, axis=1)
    conditional_variances = np.sum(centered ** 2, axis=0) / (n - 1)
    floor_raw = float(conditional_variances.mean())
    floor = floor_raw * weights.mean()
    adjusted = mse - floor_raw * weights[:, None]

    # Removing a realization changes its score, its scale, AND the common
    # sample variance. The centered identity avoids subtracting large raw means.
    floor_raw_loo = (row_squares.sum() - n / (n - 1) * row_squares) / (n - 2)
    weight_loo = (weights.sum() - weights) / (n - 1)
    floor_loo = floor_raw_loo * weight_loo
    mse_loo = (mse.sum(axis=0) - mse) / (n - 1)
    excess_loo = mse_loo - floor_loo[:, None]
    joint_loo = np.column_stack([mse_loo, excess_loo, floor_loo])
    deviations = joint_loo - joint_loo.mean(axis=0)
    covariance = (n - 1) / n * (deviations.T @ deviations)
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    n_methods = len(method_names)
    critical = t.ppf(.975, n - 1)
    summary = pd.DataFrame({
        "method": method_names, "n_samples": n,
        "mse_mean": mse.mean(axis=0), "mse_se": mse.std(axis=0, ddof=1) / np.sqrt(n),
        "noise_floor_mean": floor, "noise_floor_se": se[-1],
        "excess_mean": adjusted.mean(axis=0), "excess_se": se[n_methods:2*n_methods],
    })
    summary["excess_ci_low"] = summary.excess_mean - critical * summary.excess_se
    summary["excess_ci_high"] = summary.excess_mean + critical * summary.excess_se
    labels = (["mse_" + m for m in method_names] +
              ["excess_" + m for m in method_names] + ["noise_floor"])
    return dict(summary=summary, adjusted_by_sample=adjusted,
                conditional_variances=conditional_variances,
                floor_raw=floor_raw, floor_by_sample=floor_raw * weights,
                jackknife=pd.DataFrame(joint_loo, columns=labels),
                jackknife_covariance=pd.DataFrame(covariance, index=labels, columns=labels))


def analyze_saved_run(results, data_dir=None):
    """Validate CSV hashes, fixed covariates/splits/scales, then score correction.

    This loader deliberately requires complete train/test rows. Reduced quick
    and scalability runs are not mixed into the manuscript's evaluation.
    """
    results = results.sort_values("sample").reset_index(drop=True)
    if len(results) < 3 or results["sample"].duplicated().any():
        raise ValueError("Need at least three distinct scored dataset samples.")
    for key in ["dataset", "seed", "test_size", "scale_y_tau"]:
        if results[key].nunique(dropna=False) != 1:
            raise ValueError(f"All realizations must share {key}.")
    effects, source_files = [], []
    reference_x = None
    for row in results.itertuples(index=False):
        path = Path(data_dir or DEFAULT_DATA_DIR) / "lalonde_cps.zip"
        raw = read_sample_bytes(row.dataset, int(row.sample), data_dir)
        sha = hashlib.sha256(raw).hexdigest()
        if sha != row.source_sha256:
            raise ValueError(f"Scored data changed: {path.name}")
        x, _, y, delta = map_df_to_XAYtau(pd.read_csv(io.BytesIO(raw)))
        if reference_x is None:
            reference_x = x
        elif not np.array_equal(reference_x, x):
            raise ValueError("Noise correction requires identical covariates and row order.")
        train, test = train_test_split(np.arange(len(x)), test_size=float(row.test_size),
                                      random_state=int(row.seed), shuffle=True, stratify=None)
        if (len(train), len(test)) != (int(row.n_train), int(row.n_test)):
            raise ValueError("Noise-corrected manuscript scoring requires full train/test rows.")
        for indices, key in [(train, "train_rows_sha256"), (test, "test_rows_sha256")]:
            if hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest() != getattr(row, key):
                raise ValueError(f"Scoring split mismatch: {path.name}, {key}")
        scale = float(y[train].max() - y[train].min()) if row.scale_y_tau else 1.0
        if not np.isclose(scale, row.y_train_range_raw, rtol=1e-12, atol=1e-12):
            raise ValueError(f"Training outcome scale mismatch: {path.name}")
        if not row.target_labels_excluded_from_X or row.target != "sampled_y1_minus_y0":
            raise ValueError("Saved results must exclude effect labels and score sampled effects.")
        effects.append(delta[test])
        source_files.append(dict(sample=int(row.sample), path=f"data/realcause/lalonde_cps.zip!{row.dataset}_sample{int(row.sample)}.csv", sha256=sha))
    report = noise_adjusted_statistics(
        results[list(METHOD_COLUMNS.values())].to_numpy(), np.stack(effects),
        results.y_train_range_raw.to_numpy(), list(METHOD_COLUMNS))
    per_sample = results[["sample", "y_train_range_raw", *METHOD_COLUMNS.values()]].copy()
    per_sample["estimated_noise_floor"] = report["floor_by_sample"]
    for j, method in enumerate(METHOD_COLUMNS):
        per_sample[METHOD_COLUMNS[method].replace("mse_", "excess_")] = report["adjusted_by_sample"][:, j]
    report["per_sample"] = per_sample
    report["jackknife"].insert(0, "deleted_sample", results["sample"].to_numpy())
    report["conditional_variances"] = pd.DataFrame({
        "test_row": test, "sampled_effect_variance_raw": report["conditional_variances"]})
    report["metadata"] = dict(
        n_realizations=len(results), scored_sample_ids=results["sample"].astype(int).tolist(),
        noise_reference_sample_ids=results["sample"].astype(int).tolist(),
        n_test_rows=len(test), seed=int(results.seed.iloc[0]),
        test_rows_sha256=results.test_rows_sha256.iloc[0],
        identical_covariates_verified=True, source_files=source_files,
        raw_noise_floor=report["floor_raw"],
        normalized_noise_floor_mean=float(report["summary"].noise_floor_mean.iloc[0]),
        normalized_noise_floor_joint_jackknife_se=float(report["summary"].noise_floor_se.iloc[0]),
        variance_ddof=1, uncertainty="joint delete-one-realization jackknife; t_(R-1) approximate 95% CI",
        estimator_refits=0, clipped_negative_estimates=False,
        interpretation="Estimated normalized CATE risk; conditional on the fitted generator, empirical covariates and fixed split.",
    )
    return report


def save_noise_report(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for key, name in [("summary", "summary"), ("per_sample", "samples"),
                      ("conditional_variances", "test_row_variances"), ("jackknife", "jackknife")]:
        report[key].to_csv(directory / f"realcause_noise_{name}.csv", index=False)
    report["jackknife_covariance"].to_csv(directory / "realcause_noise_covariance.csv")
    (directory / "realcause_noise_manifest.json").write_text(
        json.dumps(report["metadata"], indent=2) + "\n")


def plot_noise_comparison(summary):
    """Original risk and estimated excess risk; matched methods, linear axes."""
    import matplotlib.pyplot as plt
    colors = ["#0072B2", "#80858F", "#009E73", "#CC79A7"]
    n = int(summary.n_samples.iloc[0])
    critical = t.ppf(.975, n - 1)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), layout="constrained")
        for ax, prefix, title in zip(axes, ["mse", "excess"],
                                    ["(a) Sampled-effect MSE", "(b) Estimated excess risk"]):
            means = summary[prefix + "_mean"].to_numpy()
            errors = critical * summary[prefix + "_se"].to_numpy()
            for j, (mean, error, color) in enumerate(zip(means, errors, colors)):
                ax.errorbar(j, mean, yerr=error, fmt="o", color=color,
                            markersize=5, capsize=3, linewidth=1.2)
            ax.set_xticks(range(len(summary)), summary.method)
            ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
            ax.set_ylabel("Normalized squared error")
            lower = min(0.0, float((means-errors).min()) * 1.1)
            ax.set_ylim(lower, float((means+errors).max()) * 1.13)
            ax.set_xlim(-.45, len(summary)-.55)
            ax.grid(axis="y", alpha=.2, linewidth=.6)
        floor = float(summary.noise_floor_mean.iloc[0])
        axes[0].axhline(floor, color="#555555", linestyle="--", linewidth=.9,
                       label=f"Estimated noise floor: {floor:.5f}")
        axes[0].legend(loc="upper left", fontsize=7.4, frameon=False)
    return fig


def latex_realcause_section(report, results, paired_stats):
    """Report the same-realization noise correction and method comparison."""
    for key, expected in {'backend': 'nystrom', 'nystrom_m': 700,
                          'nystrom_random_state': 700, 'seed': 50, 'test_size': .5}.items():
        if not results[key].eq(expected).all():
            raise ValueError(f'The paper template requires {key}={expected}.')
    summary = report['summary'].set_index('method')
    n = len(results)
    first = results.iloc[0]
    floor = summary.noise_floor_mean.iloc[0]
    floor_se = summary.noise_floor_se.iloc[0]
    ours = summary.loc['Ours', 'excess_mean']
    ids = f"${int(results['sample'].min())}$--${int(results['sample'].max())}$"
    text = r"""\subsection{Semi-real Data (RealCause)}\label{sec:realcause-experiments}
We use R_COUNT realizations of \texttt{lalonde\_cps} from RealCause \citep{neal2020realcause} (sample IDs SAMPLE_IDS). Each has N_TOTAL observations and eight covariates. All methods use only factual $(X,A,Y)$ for fitting, with a common $50/50$ train/test split (seed $50$; N_TRAIN training rows). Outcome scaling is fitted on training data; effects are divided by the training outcome range $a_r$. The nuisance and target Mat\'ern kernels use $(\nu,\ell)=(1.5,1.3)$ and $(2.5,1.2)$, respectively, with the ridge grid and fitting schemes above. True propensities are unavailable, so oracle DR is omitted.

All methods use Nystr\"om KRR with at most 700 uniformly sampled training landmarks per fit (seed $700$); smaller fits use all training points. For $N$ fitting rows and $m$ landmarks, storage is $O(Nm+m^2)$ and ridge-solve cost is $O(Nm^2+m^3)$. Runs use macOS/ARM64 with one BLAS thread per file; software settings and elapsed times are supplied with the code. The theoretical results concern exact KRR.

\paragraph{Evaluation.}
Let $\Delta_{ir}=Y_{ir}(1)-Y_{ir}(0)$ be the sampled effect at test covariate $x_i$ in realization $r$. We report normalized sampled-effect MSE,
\[
M_{mr}=\frac{1}{n_T}\sum_i\left(\widehat h_{mr}(x_i)-\Delta_{ir}/a_r\right)^2.
\]
Its expectation is CATE estimation error plus a common conditional-variance term. Since the realizations share covariates and test rows, we estimate this term using the same $R=R_COUNT$ files:
\begin{equation}\label{eq:realcause-excess-risk}
\widehat B=\frac{1}{n_T}\sum_i\widehat{\operatorname{Var}}_{r}(\Delta_{ir}),\qquad
\widehat C=\widehat B\frac{1}{R}\sum_r a_r^{-2},\qquad
\widehat E_m=\frac{1}{R}\sum_r M_{mr}-\widehat C.
\end{equation}
The sample variances use denominator $R-1$. The common correction preserves method rankings and paired differences. Excess-risk SEs use a joint delete-one-realization jackknife, recomputing both the score average and the correction after each deletion; MSE SEs are standard deviations divided by $\sqrt R$. This accounts for reuse of the same files, conditional on the fitted generator, covariates, and split.

"""
    for key, value in {'R_COUNT': str(n), 'SAMPLE_IDS': ids,
                       'N_TOTAL': f'{int(first.n_total):,}', 'N_TRAIN': f'{int(first.n_train):,}'}.items():
        text = text.replace(key, value)
    text += f'The estimated noise floor is ${floor:.5f}$ (SE ${floor_se:.5f}$). '
    if summary.excess_mean.idxmin() == 'Ours' and (summary.excess_mean > 0).all():
        reductions = [100 * (1 - ours / summary.loc[m, 'excess_mean']) for m in ['Plug-in', 'DR', 'R']]
        text += (r'Table~\ref{table:realcause} shows the lowest mean error for Ours; '
                 f'its estimated excess risk is ${reductions[0]:.1f}\\%$, '
                 f'${reductions[1]:.1f}\\%$, and ${reductions[2]:.1f}\\%$ '
                 'below plug-in, DR, and R, respectively.\n\n')
    else:
        text += f'{summary.excess_mean.idxmin()} has the lowest mean estimated excess risk.\n\n'
    text += r"""\begin{table}[htbp]
\centering
\caption{RealCause results on R_COUNT realizations. Entries are mean normalized error (SE). Estimated excess risk subtracts a common noise floor estimated from the same files; its SE includes correction uncertainty through the joint jackknife.}
\label{table:realcause}
\begin{tabular}{@{}lcc@{}}
\toprule
Method & Sampled-effect MSE (SE) & Estimated excess risk (SE) \\
\midrule
""".replace('R_COUNT', str(n))
    for method, row in summary.iterrows():
        values = []
        for metric in ['mse', 'excess']:
            mean = f'{row[metric+"_mean"]:.5f}'
            if method == summary[metric+'_mean'].idxmin():
                mean = r'\textbf{' + mean + '}'
            values.append(mean + f' ({row[metric+"_se"]:.5f})')
        text += method + ' & ' + ' & '.join(values) + r' \\' + '\n'
    return text + '\\bottomrule\n\\end{tabular}\n\\end{table}\n\\clearpage\n\n'
