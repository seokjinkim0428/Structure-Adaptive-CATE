"""RealCause settings, execution, and summaries."""

from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import hashlib
import json
from numbers import Integral
import platform
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from ours.realcause import run_one_dataset_sample, DEFAULT_DATA_DIR, REALCAUSE_NYSTROM_M

METHOD_COLUMNS = {
    "Ours": "mse_ours",
    "Plug-in": "mse_plugin",
    "DR": "mse_dr",
    "R": "mse_r",
}


def experiment_config(profile="full", n_samples=None, start_sample=0):
    """Select consecutive files (IDs 0–99); full uses every row in each file."""
    if profile not in {"quick", "full"}:
        raise ValueError("profile must be 'quick' or 'full'.")
    if isinstance(start_sample, bool) or not isinstance(start_sample, Integral) or not 0 <= start_sample < 100:
        raise ValueError("start_sample must be an integer from 0 through 99.")
    if n_samples is None:
        n_samples = min(2 if profile == "quick" else 50, 100 - start_sample)
    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer.")
    if start_sample + n_samples > 100:
        raise ValueError(
            f"Requested {n_samples} samples starting at {start_sample}, "
            f"but only {100 - start_sample} files remain (IDs 0..99)."
        )
    return {
        "profile": profile,
        "dataset": "lalonde_cps",
        "samples": list(range(int(start_sample), int(start_sample + n_samples))),
        "test_size": 0.5,
        "seed": 50,
        "scale_y_tau": True,
        "xi": 0.01,
        "lams": [1, 8, 64] if profile == "quick" else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        "kf_length": 1.3,
        "kf_nu": 1.5,
        "kh_length": 1.2,
        "kh_nu": 2.5,
        "krr_kwargs": {"nystrom_m": REALCAUSE_NYSTROM_M, "nystrom_random_state": 700},
        "max_train_size": 1200 if profile == "quick" else None,
        "max_test_size": 1200 if profile == "quick" else None,
        "blas_threads": 1,
    }


def summarize_results(results):
    """Mean and Monte Carlo SE across generated dataset samples, not test rows."""
    rows = []
    for method, column in METHOD_COLUMNS.items():
        values = results[column].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite results for {method}; refusing to drop failed runs.")
        n = len(values)
        std = float(values.std(ddof=1)) if n >= 2 else np.nan
        key = column.removeprefix("mse_")
        rows.append({
            "method": method, "mse_mean": float(values.mean()) if n else np.nan,
            "mse_std": std, "mse_se": std / np.sqrt(n) if n >= 2 else np.nan,
            "n_samples": n,
            "mean_seconds": float(results[f"seconds_{key}"].mean()) if n else np.nan,
        })
    return pd.DataFrame(rows)


def run_samples(config, data_dir=None, checkpoint_path=None, verbose=True, resume=True):
    """Run a profile, saving completed samples after every successful sample."""
    kwargs = {k: v for k, v in config.items() if k not in {"profile", "samples", "blas_threads"}}
    rows = []
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = checkpoint_path.parent / "manifest.json"
        manifest = make_run_manifest(config, data_dir)
        if resume and checkpoint_path.exists():
            if not manifest_path.exists():
                raise ValueError("Checkpoint lacks its run manifest.")
            previous = json.loads(manifest_path.read_text())
            for key in ["config", "source_sha256", "dependencies"]:
                if previous[key] != manifest[key]:
                    raise ValueError(f"Checkpoint {key} differs; use a new output directory.")
            rows = pd.read_csv(checkpoint_path).to_dict("records")
            ids = [int(row["sample"]) for row in rows]
            if ids != config["samples"][:len(ids)]:
                raise ValueError("Checkpoint samples are not a valid prefix of this run.")
            if any(row["backend"] != "nystrom" for row in rows):
                raise ValueError("Checkpoint is not a Nyström run.")
        else:
            save_manifest(manifest_path, manifest)
    completed = {int(row["sample"]) for row in rows}
    with threadpool_limits(limits=config.get("blas_threads", 1)):
        for index, sample in enumerate(config["samples"], start=1):
            if sample in completed:
                continue
            if verbose:
                print(f"[{index}/{len(config['samples'])}] RealCause sample file {sample}", flush=True)
            result = run_one_dataset_sample(
                sample=int(sample), data_dir=data_dir, verbose=verbose, **kwargs,
            )
            result["profile"] = config["profile"]
            for key in METHOD_COLUMNS.values():
                result[f"{key}_raw"] = result[key] * result["y_train_range_raw"] ** 2
            rows.append(result)
            if checkpoint_path is not None:
                temporary = checkpoint_path.with_suffix(".tmp")
                pd.DataFrame(rows).to_csv(temporary, index=False)
                temporary.replace(checkpoint_path)
            if verbose:
                scores = ", ".join(f"{m}={result[c]:.6f}" for m, c in METHOD_COLUMNS.items())
                print(f"sample {sample}: {scores}; {result['seconds_total']:.2f} seconds")
    return pd.DataFrame(rows)


def make_run_manifest(config, data_dir=None, extra=None):
    """Capture environment and scientific settings without importing model trainers."""
    dependencies = {}
    for package in ["numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "nbformat", "threadpoolctl"]:
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = "not installed"
    root = Path(__file__).resolve().parent
    source_hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ["realcause.py", "realcause_experiments.py", "kernels.py",
                     "krr.py", "model_selection.py", "imputation.py",
                     "plugin.py", "dr.py", "rlearner.py"]
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "data_dir": "data/realcause",
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": dependencies,
        "threadpools_available": threadpool_info(),
        "source_sha256": source_hashes,
        "metric": "mean squared error against supplied sampled (y1-y0) / training outcome range",
        "data_and_metric_notes": [
            "Eight covariates; outcome and effect labels are excluded.",
            "Nyström KRR with up to 700 training landmarks per fit.",
            "SE across files uses ddof=1 and requires at least two files.",
        ],
    }
    if extra:
        manifest.update(extra)
    return manifest


def save_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return path
