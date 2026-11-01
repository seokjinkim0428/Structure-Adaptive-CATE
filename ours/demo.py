"""Notebook settings and result displays."""

from dataclasses import replace
import hashlib
import json
from numbers import Integral
from pathlib import Path

from ours.synthetic_experiments import CORE_GROUPS, build_conditions, result_table
from ours.synthetic_reporting import comparison_table
from ours.realcause_experiments import make_run_manifest, summarize_results


STUDIES = {
    "all": CORE_GROUPS,
    "sample_size": ("original",),
    "noise": ("noise",),
    "overlap": ("kappa", "overlap_extension"),
    "nuisance": ("nuisance",),
    "representation": ("representation",),
    "crossfit": ("crossfit",),
    "undersmoothing": ("undersmoothing",),
    "same_kernel": ("same_kernel",),
}


def select_synthetic_conditions(profile, setting="all", study="all", n_replications=None):
    """Select a study and retain matched controls for the single-split comparison."""
    if setting not in ("all", "univariate", "multivariate"):
        raise ValueError("setting must be 'all', 'univariate', or 'multivariate'")
    if study not in STUDIES:
        raise ValueError(f"study must be one of {tuple(STUDIES)}")
    if n_replications is not None and (
        isinstance(n_replications, bool) or not isinstance(n_replications, Integral)
        or n_replications < 1
    ):
        raise ValueError("n_replications must be a positive integer or None")
    groups = STUDIES[study]
    available = build_conditions(profile, groups=tuple(set(groups) | {"noise"}))
    selected = [c for c in available if c.group in groups]
    if setting != "all":
        selected = [c for c in selected if (c.d == 1) == (setting == "univariate")]
    controls = {f"noise__d{c.d}_p{c.p}_sigma1" for c in selected if c.group == "crossfit"}
    ids = {c.condition_id for c in selected}
    selected += [c for c in available if c.condition_id in controls - ids]
    if not selected:
        raise ValueError(f"No {setting} conditions in the {study} study")
    if n_replications is not None:
        selected = [replace(c, n_replications=int(n_replications)) for c in selected]
    return selected


def show_synthetic_results(summary, output_dir, plots=False):
    from IPython.display import Markdown, display
    import matplotlib.pyplot as plt
    from ours.synthetic_reporting import plot_univariate_comparison, plot_multivariate_comparison

    output_dir = Path(output_dir)
    for name, plotter in (("univariate", plot_univariate_comparison),
                          ("multivariate", plot_multivariate_comparison)):
        part = summary[summary.d.eq(1) if name == "univariate" else summary.d.gt(1)]
        if part.empty:
            continue
        table = comparison_table(part, name)
        display(Markdown(f"### {name.capitalize()}"))
        if not table.empty:
            table.to_csv(output_dir / f"{name}.csv", index=False)
            display(table.style.hide(axis="index"))
        for study in ("undersmoothing", "same_kernel"):
            extra = result_table(part, study)
            if not extra.empty:
                extra.to_csv(output_dir / f"{name}_{study}.csv")
                display(extra)
        if plots and not table.empty:
            fig = plotter(part)
            fig.savefig(output_dir / f"{name}.png", dpi=180, bbox_inches="tight")
            plt.show()


def realcause_run_dir(config, output_dir="outputs/realcause"):
    """Keep checkpoints for different settings and code versions separate."""
    manifest = make_run_manifest(config)
    identity = {k: manifest[k] for k in ("config", "source_sha256", "dependencies")}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    samples = config["samples"]
    return Path(output_dir) / f"{config['profile']}_s{samples[0]}_n{len(samples)}" / key


def show_realcause_results(results, output_dir, profile, plots=False):
    from IPython.display import display
    import matplotlib.pyplot as plt
    from ours.realcause_reporting import analyze_saved_run, save_noise_report, plot_noise_comparison

    output_dir = Path(output_dir)
    summary = summarize_results(results)
    summary.to_csv(output_dir / "summary.csv", index=False)
    if profile == "full" and len(results) >= 3:
        report = analyze_saved_run(results)
        save_noise_report(report, output_dir)
        table = report["summary"][["method", "mse_mean", "mse_se", "excess_mean", "excess_se"]]
        if plots:
            fig = plot_noise_comparison(report["summary"])
    else:
        table = summary[["method", "mse_mean", "mse_se"]]
        if plots:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.errorbar(table.method, table.mse_mean, yerr=table.mse_se.fillna(0), fmt="o", capsize=4)
            ax.set_ylabel("Sampled-effect MSE (±1 SE)")
    display(table.style.hide(axis="index").format(precision=5))
    if plots:
        fig.savefig(output_dir / "comparison.png", dpi=180, bbox_inches="tight")
        plt.show()
    return table
