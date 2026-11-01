"""Synthetic experiment settings, execution, and summaries."""

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

import ours.synthetic as synthetic
from ours.kernels import get_median_based_length_scales
from ours.imputation import fit_ours
from ours.plugin import fit_plugin
from ours.dr import fit_dr
from ours.rlearner import fit_r_learner


METHODS = ("Ours", "Plug-in", "DR", "Oracle DR", "R")
CORE_GROUPS = (
    "original", "noise", "kappa", "representation", "nuisance",
    "overlap_extension", "crossfit",
)
OPTIONAL_GROUPS = ("undersmoothing", "same_kernel")
RIDGE_GRID = tuple(float(2 ** j) for j in range(10))
SEED_BASE = 100
SEED_STRIDE = 10
TEST_SEED = 10099
LENGTH_SEED = 42
KRR_KWARGS = {"jitter": 1e-8, "lstsq_fallback": True, "fallback_jitter": 1e-6}

# Accept labels in earlier result files.
LEGACY_METHOD_LABELS = {
    "UCATE": "Ours",
    "UCATE single split": "Ours single split",
    "UCATE H=F": "Ours H=F",
}


def canonical_method_name(method):
    return LEGACY_METHOD_LABELS.get(method, method)


def normalize_method_names(frame):
    """Return a copy with current method labels, preserving every numeric value."""
    frame = frame.copy()
    frame["method"] = frame["method"].map(canonical_method_name)
    return frame


@dataclass(frozen=True)
class Condition:
    condition_id: str
    group: str
    provenance: str
    label: str
    d: int
    p: int
    n_samples: int
    noise_std: float = 1.0
    kappa: float = 0.1
    xi: float = 0.01
    n_replications: int = 30
    n_test: int = 3000
    kernel_recipe: str = "median"
    subsets: tuple = ("all", "x1-4")
    nuisance_order: int = 1
    target_order: int = 2
    lambda_grid: tuple = RIDGE_GRID
    methods: tuple = METHODS


def _condition(group, label, d, p, n, **kwargs):
    token = label.lower().replace(" ", "_").replace("=", "").replace("+", "_")
    return Condition(
        condition_id=f"{group}__{token}", group=group, label=label,
        provenance=kwargs.pop("provenance", "sensitivity experiment"),
        d=d, p=p, n_samples=n, **kwargs,
    )


def build_conditions(profile="quick", groups=None):
    """Paper settings (full) or a reduced two-replication run (quick)."""
    if profile not in ("quick", "full"):
        raise ValueError("profile must be 'quick' or 'full'")
    conditions = []
    for d, p, ns in ((1, 1, (500, 1000)), (10, 10, (1000, 2000)), (10, 4, (1000, 2000))):
        for n in ns:
            conditions.append(_condition(
                "original", f"d{d} p{p} n{n}", d, p, n,
                n_replications=100, kernel_recipe="legacy",
                methods=METHODS, provenance="sample-size experiment",
            ))

    cases = ((1, 1, 500), (10, 10, 1000), (10, 4, 1000), (20, 20, 1000), (20, 4, 1000))
    for d, p, n in cases:
        for sigma in ((0.25, 0.5, 1.0, 2.0, 4.0) if d == 1 else (0.5, 1.0, 2.0)):
            conditions.append(_condition("noise", f"d{d} p{p} sigma{sigma:g}", d, p, n, noise_std=sigma))
        for kappa in ((0.02, 0.05, 0.1, 0.2, 0.3) if d == 1 else (0.05, 0.1, 0.2)):
            conditions.append(_condition("kappa", f"d{d} p{p} kappa{kappa:g}", d, p, n, kappa=kappa))
        for kappa in (0.01, 0.5):
            conditions.append(_condition(
                "overlap_extension", f"d{d} p{p} kappa{kappa:g}", d, p, n, kappa=kappa,
                provenance="overlap sweep",
            ))
        conditions.append(_condition(
            "crossfit", f"d{d} p{p}", d, p, n,
            methods=("Ours", "Ours single split"),
            provenance="cyclic averaging comparison",
        ))
        conditions.append(_condition(
            "same_kernel", f"d{d} p{p}", d, p, n,
            methods=("Ours", "Ours H=F"),
            provenance="target-kernel comparison",
        ))

    for name, subsets in (
        ("correct", ("all", "x1-4")),
        ("correct plus wrong", ("all", "x1-4", "x5-8")),
        ("wrong only", ("x5-8",)),
    ):
        conditions.append(_condition("representation", name, 20, 4, 1000, subsets=subsets))
    for name, f_order, h_order in (("H1 H2", 1, 2), ("H2 H3", 2, 3)):
        conditions.append(_condition(
            "nuisance", name, 1, 1, 1000,
            nuisance_order=f_order, target_order=h_order,
        ))
    for d, p, n in ((1, 1, 500), (10, 10, 1000), (20, 4, 1000)):
        for sigma in (0.5, 1.0, 2.0):
            for c in (0.001, 0.01, 0.1, 1.0):
                conditions.append(_condition(
                    "undersmoothing", f"d{d} p{p} sigma{sigma:g} c{c:g}", d, p, n,
                    noise_std=sigma, xi=c, methods=("Ours",),
                    provenance="nuisance regularization sweep",
                ))

    if profile == "quick":
        def keep(c):
            if c.group == "original":
                return c.n_samples == (500 if c.d == 1 else 1000)
            if c.group == "noise":
                return c.noise_std == 1.0
            if c.group == "kappa":
                return (c.d, c.p) in ((1, 1), (10, 10), (20, 4)) and c.kappa in (0.05, 0.2)
            if c.group == "overlap_extension":
                return c.d == 1
            if c.group in ("crossfit", "same_kernel"):
                return (c.d, c.p) in ((1, 1), (20, 4))
            if c.group == "undersmoothing":
                return c.noise_std == 1.0 and c.xi in (0.001, 0.1)
            return True
        conditions = [replace(c, n_samples=120, n_replications=2, n_test=300,
                              lambda_grid=(1.0, 8.0, 64.0)) for c in conditions if keep(c)]
    if groups is not None:
        groups = tuple(groups)
        unknown = set(groups) - set(CORE_GROUPS + OPTIONAL_GROUPS)
        if unknown:
            raise ValueError(f"Unknown experiment groups: {sorted(unknown)}")
        conditions = [c for c in conditions if c.group in groups]
    return conditions


def design_table(conditions):
    """Condition settings as a DataFrame."""
    return pd.DataFrame([{
        "condition_id": c.condition_id, "study": c.group, "d": c.d, "p": c.p,
        "n": c.n_samples, "sigma": c.noise_std, "kappa": c.kappa, "c": c.xi,
        "replications": c.n_replications, "test_points": c.n_test,
        "ridge_candidates": len(c.lambda_grid), "provenance": c.provenance,
    } for c in conditions])


@lru_cache(maxsize=None)
def calibrated_lengths(d):
    # The extra x5:8 support does not affect lengths for all or x1:4.
    # Calibration uses independent synthetic covariates, never test outcomes.
    return get_median_based_length_scales(
        d=d, subset_modes=("all", "x1-4", "x5-8"),
        nu_candidates=(1.5, 2.5), include_rbf=True,
        n_sample=2000, random_seed=LENGTH_SEED,
    )


def make_kernels(condition):
    c = condition
    if c.d == 1:
        lengths = 1.0
        length_f = 1.0
    elif c.kernel_recipe == "legacy":
        lengths = {
            ("all", 1.5): 2.6, ("all", 2.5): 2.4,
            ("x1-4", 1.5): 1.6, ("x1-4", 2.5): 1.5,
            ("all", "rbf"): 2.1, ("x1-4", "rbf"): 1.3,
        }
        length_f = 2.6
    else:
        lengths = dict(calibrated_lengths(c.d))
        length_f = lengths[("all", 1.5)]
    kernels = synthetic.make_synthetic_method_kernels(
        case_type="Univariate" if c.d == 1 else "Multi-dim",
        d=c.d, p_dim=None if c.d == 1 else c.p,
        length_scale_F=length_f, nu_F=1.5, length_scale_H=lengths,
        nu_H_candidates=(1.5, 2.5), subset_modes=c.subsets,
        include_rbf=True, nu_H_for_DR=2.5,
        univariate_F_alpha=c.nuisance_order,
        univariate_H_alpha=c.target_order,
    )
    description = {
        "nuisance_length": length_f,
        "target_lengths": {str(k): v for k, v in lengths.items()} if isinstance(lengths, dict) else lengths,
        "candidate_names": [name for name, _ in kernels[1]],
        "baseline_target_support": "all" if c.p == c.d else f"x1:{c.p}",
    }
    return kernels, description


def _evaluate_method(c, method, X, A, Y, X_test, h_star, pi_func, kernels, seed):
    K_F, K_candidates, K_DR, K_R = kernels
    selected_kernels, selected_ridges = [], []
    if method in ("Ours", "Ours H=F", "Ours single split"):
        candidate_library = [("same_as_nuisance", K_F)] if method == "Ours H=F" else K_candidates
        predictor, selected = fit_ours(
            X, A, Y, c.xi, c.lambda_grid, K_F, candidate_library, seed,
            crossfit=method != "Ours single split", krr_kwargs=KRR_KWARGS,
        )
        selected_kernels = [key[0] for key in selected]
        selected_ridges = [key[1] for key in selected]
    else:
        if method == "Plug-in":
            predictor, _ = fit_plugin(X, A, Y, c.lambda_grid, K_F, seed, krr_kwargs=KRR_KWARGS)
        elif method in ("DR", "Oracle DR"):
            predictor = fit_dr(
                X, A, Y, seed, K_F, K_DR, c.lambda_grid, krr_kwargs=KRR_KWARGS,
                pi_func=pi_func if method == "Oracle DR" else None,
            )
        elif method == "R":
            predictor = fit_r_learner(X, A, Y, seed, K_F, K_R, c.lambda_grid, krr_kwargs=KRR_KWARGS)
        else:
            raise ValueError(f"Unknown method: {method}")
    mse = synthetic.calculate_mse(predictor, h_star, X_test)
    return float(mse), selected_kernels, selected_ridges


def _manifest(profile, conditions):
    root = Path(__file__).resolve().parent
    return {
        "schema_version": 1, "profile": profile,
        "conditions": [asdict(c) for c in conditions],
        "seeds": {"base": SEED_BASE, "stride": SEED_STRIDE, "test": TEST_SEED, "length": LENGTH_SEED},
        "python": platform.python_version(), "platform": platform.platform(),
        "numerical_threadpools": [{k: pool.get(k) for k in
                                  ("internal_api", "prefix", "version", "num_threads", "architecture")}
                                 for pool in threadpool_info()],
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "pandas")},
        "code_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                        for name in ("synthetic.py", "synthetic_experiments.py", "kernels.py",
                                     "krr.py", "model_selection.py", "imputation.py",
                                     "plugin.py", "dr.py", "rlearner.py")},
        "kernel_settings": {c.condition_id: make_kernels(c)[1] for c in conditions},
        "regularization": "raw Gram diagonal xi=c; normalized nuisance lambda=c/n_fit",
        "metric": "mean squared CATE error over one fixed independent uniform test sample",
        "se": "sample standard deviation across replications / sqrt(n_eff); undefined for n_eff<2",
    }


@threadpool_limits.wrap(limits=1)
def run_suite(profile="quick", groups=None, output_dir="outputs/synthetic", *,
              conditions=None, progress=True, resume=True):
    """Run conditions and save each replication, including failures.

    Matching runs resume from checkpoints. Set resume=False to recompute.
    """
    if profile not in ("quick", "full"):
        raise ValueError("profile must be 'quick' or 'full'")
    if conditions is None:
        conditions = build_conditions(profile, CORE_GROUPS if groups is None else groups)
    else:
        conditions = list(conditions)
        if groups is not None:
            raise ValueError("Specify conditions or groups, not both")
    if not conditions:
        raise ValueError("No experiment conditions selected")
    if len({c.condition_id for c in conditions}) != len(conditions):
        raise ValueError("Condition identifiers must be unique")
    manifest = _manifest(profile, conditions)
    manifest_text = json.dumps(manifest, sort_keys=True, indent=2)
    fingerprint = hashlib.sha256(manifest_text.encode()).hexdigest()[:16]
    run_dir = Path(output_dir) / profile / fingerprint
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(manifest_text + "\n")
    raw_path = run_dir / "replications.csv"
    rows = pd.read_csv(raw_path, keep_default_na=False).to_dict("records") if resume and raw_path.exists() else []
    completed = {(r["condition_id"], int(r["replication"]), r["method"]) for r in rows}
    if progress:
        print(f"{profile.upper()}: {len(conditions)} conditions. Results: {run_dir}", flush=True)
    for condition_index, c in enumerate(conditions, 1):
        if progress:
            print(f"[{condition_index}/{len(conditions)}] {c.group}: {c.label} "
                  f"(n={c.n_samples}, R={c.n_replications})", flush=True)
        kernels, _ = make_kernels(c)
        domain = (0, 1) if c.d == 1 else (-1, 1)
        f0, _, h_star = synthetic.get_true_functions(c.d, p=None if c.d == 1 else c.p,
                                                    case_type="Univariate" if c.d == 1 else "Multi-dim")
        pi_func = synthetic.make_kappa_propensity(c.kappa)
        X_test = np.random.RandomState(TEST_SEED).uniform(*domain, size=(c.n_test, c.d))
        for replication in range(c.n_replications):
            pending = [m for m in c.methods if (c.condition_id, replication, m) not in completed]
            if not pending:
                continue
            seed = SEED_BASE + SEED_STRIDE * replication
            X, A, Y, _, _, _, pi = synthetic.generate_data(
                c.n_samples, c.d, f0, h_star, pi_func, c.noise_std, seed, domain,
            )
            for method in pending:
                started = time.perf_counter()
                status, error = "ok", ""
                try:
                    mse, selected_kernels, selected_ridges = _evaluate_method(
                        c, method, X, A, Y, X_test, h_star, pi_func, kernels, seed,
                    )
                    if not np.isfinite(mse):
                        status, error = "nonfinite", "The estimator returned non-finite MISE"
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    mse, selected_kernels, selected_ridges = np.nan, [], []
                    status, error = "failed", f"{type(exc).__name__}: {exc}"
                rows.append({
                    "profile": profile, "condition_id": c.condition_id, "study": c.group,
                    "label": c.label, "provenance": c.provenance,
                    "d": c.d, "p": c.p, "n_samples": c.n_samples,
                    "noise_std": c.noise_std, "kappa": c.kappa, "c": c.xi,
                    "n_test": c.n_test, "n_requested": c.n_replications,
                    "replication": replication, "seed": seed, "test_seed": TEST_SEED,
                    "method": method, "mise": mse, "status": status, "error": error,
                    "n_treated": int(A.sum()), "n_control": int((1 - A).sum()),
                    "empirical_kappa": float(min(pi.min(), (1 - pi).min())),
                    "selected_kernels": json.dumps(selected_kernels),
                    "selected_ridges": json.dumps(selected_ridges),
                    "seconds": time.perf_counter() - started,
                })
                completed.add((c.condition_id, replication, method))
                if status != "ok" and progress:
                    print(f"  seed={seed} {method}: {status}: {error}", flush=True)
            frame = pd.DataFrame(rows)
            temp_path = raw_path.with_suffix(".tmp")
            frame.to_csv(temp_path, index=False)
            temp_path.replace(raw_path)
    raw = pd.DataFrame(rows)
    # Numeric conversion also covers resumed CSVs with missing failure values.
    raw["mise"] = pd.to_numeric(raw["mise"], errors="coerce")
    summary = summarize_results(raw)
    summary.to_csv(run_dir / "summary.csv", index=False)
    kernel_selection_counts(raw).to_csv(run_dir / "kernel_selection_counts.csv", index=False)
    raw.attrs["run_dir"] = str(run_dir.resolve())
    raw.attrs["profile"] = profile
    if progress:
        print(f"Saved {len(raw)} replications; {int((raw.status != 'ok').sum())} non-success rows.", flush=True)
    return raw


def summarize_results(raw):
    """Mean, sample SD and SE with explicit failure/effective-replication counts."""
    raw = normalize_method_names(raw)
    keys = ["profile", "condition_id", "study", "label", "provenance", "d", "p",
            "n_samples", "noise_std", "kappa", "c", "n_test", "method"]
    records = []
    for key, group in raw.groupby(keys, dropna=False, sort=False):
        values = pd.to_numeric(group.loc[group.status == "ok", "mise"], errors="coerce")
        values = values[np.isfinite(values)]
        n = len(values)
        sd = float(values.std(ddof=1)) if n >= 2 else np.nan
        records.append(dict(zip(keys, key), mise_mean=float(values.mean()) if n else np.nan,
                            mise_std=sd, mise_se=sd / np.sqrt(n) if n >= 2 else np.nan,
                            n_eff=n, n_failed=len(group) - n, n_observed=len(group),
                            n_requested=int(group.n_requested.iloc[0])))
    return pd.DataFrame(records)


def result_table(summary, study):
    summary = normalize_method_names(summary)
    subset = summary[summary.study == study].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["mean (SE)"] = subset.apply(
        lambda r: f"{r.mise_mean:.4f} ({r.mise_se:.4f})" if r.n_eff >= 2
        else f"{r.mise_mean:.4f} (SE unavailable)", axis=1,
    )
    return subset.pivot(index="label", columns="method", values="mean (SE)")


def kernel_selection_counts(raw):
    """Each cyclic fit contributes one selection per rotation (three per run)."""
    raw = normalize_method_names(raw)
    records = []
    for _, row in raw[raw.method.str.startswith("Ours") & (raw.status == "ok")].iterrows():
        for rotation, kernel in enumerate(json.loads(row.selected_kernels)):
            records.append({"condition_id": row.condition_id, "method": row.method,
                            "rotation": rotation, "kernel": kernel})
    if not records:
        return pd.DataFrame(columns=["condition_id", "method", "kernel", "count"])
    return pd.DataFrame(records).groupby(["condition_id", "method", "kernel"],
                                        dropna=False).size().rename("count").reset_index()


def plot_sweep(summary, study, x="noise_std"):
    """One panel per setting; error bars are Monte Carlo SE, not confidence bands."""
    summary = normalize_method_names(summary)
    import matplotlib.pyplot as plt
    subset = summary[summary.study == study]
    settings = list(subset[["d", "p"]].drop_duplicates().itertuples(index=False, name=None))
    if not settings:
        return None
    fig, axes = plt.subplots(1, len(settings), figsize=(4 * len(settings), 3.5), squeeze=False)
    for ax, (d, p) in zip(axes[0], settings):
        for method, rows in subset[(subset.d == d) & (subset.p == p)].groupby("method", sort=False):
            rows = rows.sort_values(x)
            ax.errorbar(rows[x], rows.mise_mean, yerr=rows.mise_se, marker="o", capsize=3, label=method)
        ax.set(xlabel={"noise_std": "Noise standard deviation", "kappa": "Propensity clipping kappa"}.get(x, x),
               ylabel="MISE", title=f"d={d}, p={p}")
        if x == "kappa":
            ax.set_xscale("log")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{study.replace('_', ' ').title()} — {subset.profile.iloc[0]} profile")
    fig.tight_layout()
    return fig
