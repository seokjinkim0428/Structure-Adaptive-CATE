"""RealCause data preparation and one-file evaluation with Nyström KRR."""

from pathlib import Path
import zipfile
import io
import hashlib
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ours.kernels import matern_kernel
from ours.imputation import fit_ours
from ours.plugin import fit_plugin
from ours.dr import fit_dr
from ours.rlearner import fit_r_learner

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "realcause"
REALCAUSE_NYSTROM_M = 700
COMMON_KRR_KWARGS = {
    "jitter": 0.0,
    "lstsq_fallback": False,
    "fallback_jitter": 1e-6,
    "nystrom_m": REALCAUSE_NYSTROM_M,
    "nystrom_random_state": 700,
    "nystrom_jitter": 1e-10,
}


def _resolved_krr_kwargs(krr_kwargs=None):
    """Return per-run solver settings without mutating module defaults."""
    options = dict(COMMON_KRR_KWARGS)
    if krr_kwargs is not None:
        options.update(krr_kwargs)
    m = options["nystrom_m"]
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 1:
        raise ValueError("RealCause requires a positive Nyström landmark count.")
    return options


def read_sample_bytes(dataset, sample, data_dir=None):
    """Read a bundled CSV without extracting or modifying the data archive."""
    if dataset != "lalonde_cps":
        raise ValueError("The demo includes lalonde_cps only.")
    if isinstance(sample, bool) or not isinstance(sample, (int, np.integer)) or not 0 <= sample < 100:
        raise ValueError("sample must be an integer from 0 through 99.")
    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    with zipfile.ZipFile(directory / "lalonde_cps.zip") as archive:
        return archive.read(f"{dataset}_sample{int(sample)}.csv")


class MinMax01Scaler:
    """Per-feature min-max scaler to [0,1] (for X)."""
    def __init__(self):
        self.min_ = None
        self.range_ = None
    def fit(self, X):
        X = np.asarray(X, float)
        self.min_ = np.min(X, axis=0, keepdims=True)
        max_ = np.max(X, axis=0, keepdims=True)
        self.range_ = np.clip(max_ - self.min_, 1e-12, None)
        return self
    def transform(self, X):
        X = np.asarray(X, float)
        return (X - self.min_) / self.range_
    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMax01Scaler1D:
    """Scalar min-max scaler to [0,1] (for Y)."""
    def __init__(self):
        self.min_ = None
        self.range_ = None
    def fit(self, y):
        y = np.asarray(y, float).ravel()
        self.min_ = np.min(y)
        max_ = np.max(y)
        self.range_ = float(np.clip(max_ - self.min_, 1e-12, None))
        return self
    def transform(self, y):
        y = np.asarray(y, float).ravel()
        return (y - self.min_) / self.range_
    def transform_tau(self, tau):
        return np.asarray(tau, float).ravel() / self.range_


def _to_int01_treatment(a):
    if pd.api.types.is_bool_dtype(a):
        return a.astype(int).values
    s = a.astype(str).str.strip().str.lower()
    values = s.replace({"true": "1", "false": "0"}).astype(float).to_numpy()
    if not np.isin(values, [0.0, 1.0]).all():
        raise ValueError("Treatment must be binary (0 or 1).")
    return values.astype(int)


def load_realcause_df(dataset: str, sample: int, data_dir=None) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(read_sample_bytes(dataset, sample, data_dir)))


def map_df_to_XAYtau(df: pd.DataFrame):
    """Map covariates, treatment, observed outcome, and supplied effect target.

    RealCause's ``y0``/``y1`` are sampled potential outcomes, so their difference
    is an ITE target, not a noiseless conditional CATE. Effect labels must never
    enter the covariates. Prefer the supplied factual outcome if it is present.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    def first(names):
        return next((cols_lower[k] for k in names if k in cols_lower), None)

    trt_col = first(["t", "treatment", "a", "z"])
    if trt_col is None:
        raise ValueError("Treatment column (t/treatment/a/z) not found.")
    A = _to_int01_treatment(df[trt_col])
    if not np.isin(A, [0, 1]).all():
        raise ValueError("Treatment must be binary (0 or 1).")
    yf = first(["y_factual", "y", "y_obs", "yf"])
    y0_col = first(["y0", "mu0", "y_0"])
    y1_col = first(["y1", "mu1", "y_1"])
    ycf = first(["y_cfactual", "y_counterfactual", "ycf"])
    if y0_col is not None and y1_col is not None:
        y0 = df[y0_col].to_numpy(float)
        y1 = df[y1_col].to_numpy(float)
        tau = y1 - y0
        Y = df[yf].to_numpy(float) if yf is not None else A * y1 + (1 - A) * y0
    elif yf is not None and ycf is not None:
        Y = df[yf].to_numpy(float)
        y_cf = df[ycf].to_numpy(float)
        tau = (2 * A - 1) * (Y - y_cf)
    else:
        raise ValueError("Need either (y0,y1) or factual and counterfactual outcomes.")

    label_names = {
        "t", "treatment", "a", "z", "y", "y_factual", "y_obs", "yf",
        "y_cfactual", "y_counterfactual", "ycf", "mu0", "mu1", "id",
        "y0", "y1", "y_0", "y_1", "ite", "tau", "cate", "effect",
        "true_tau", "tau_true", "true_ite", "true_cate", "sample_id",
    }
    x_cols = [c for c in df.columns if c.lower().startswith("x") and c.lower() not in label_names]
    if not x_cols:
        x_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                  and c.lower() not in label_names and not c.lower().startswith("unnamed:")]
    if not x_cols:
        raise ValueError("No covariate columns found for X.")
    X = df[x_cols].to_numpy(float)
    if not all(np.isfinite(v).all() for v in [X, Y, tau]):
        raise ValueError("Covariates, outcomes, and effect targets must be finite.")
    return X, A, Y, tau


def prepare_realcause_sample(
    dataset, sample, data_dir=None, test_size=0.5, seed=50,
    scale_y_tau=True, max_train_size=None, max_test_size=None,
):
    """Split first, optionally subsample, then fit all scalers on training data.

    Size caps are only for quick demos and scalability experiments. No treatment
    stratification or outcome-dependent sampling is used, matching the source.
    Test values can be outside [0, 1]; no clipping changes the evaluation target.
    """
    df = load_realcause_df(dataset, sample, data_dir=data_dir)
    X, A, Y, tau = map_df_to_XAYtau(df)
    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=test_size, random_state=seed,
        shuffle=True, stratify=None,
    )
    # Separate RNGs make test rows identical across paired training-size runs.
    for size in [max_train_size, max_test_size]:
        if size is not None and (int(size) != size or size < 1):
            raise ValueError("Size caps must be positive integers or None.")
    if max_train_size is not None and len(train_idx) > max_train_size:
        train_idx = np.random.default_rng(seed).permutation(train_idx)[:max_train_size]
    if max_test_size is not None and len(test_idx) > max_test_size:
        test_idx = np.random.default_rng(seed + 1).permutation(test_idx)[:max_test_size]
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    x_scaler = MinMax01Scaler().fit(X_train_raw)
    X_train, X_test = x_scaler.transform(X_train_raw), x_scaler.transform(X_test_raw)
    y_scaler = MinMax01Scaler1D().fit(Y[train_idx]) if scale_y_tau else None
    y_range = y_scaler.range_ if y_scaler is not None else 1.0
    y_min = y_scaler.min_ if y_scaler is not None else 0.0
    Y_train = (Y[train_idx] - y_min) / y_range
    Y_test = (Y[test_idx] - y_min) / y_range
    tau_test = tau[test_idx] / y_range
    metadata = {
        "dataset": dataset, "sample": int(sample), "seed": int(seed),
        "n_total": int(len(X)), "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "d": int(X.shape[1]), "test_size": float(test_size),
        "n_treated_train": int(A[train_idx].sum()), "n_treated_test": int(A[test_idx].sum()),
        "y_train_min_raw": float(y_min), "y_train_range_raw": float(y_range),
        "y_train_scaled_min": float(Y_train.min()), "y_train_scaled_max": float(Y_train.max()),
        "y_test_scaled_min": float(Y_test.min()), "y_test_scaled_max": float(Y_test.max()),
        "scale_y_tau": bool(scale_y_tau), "target": "sampled_y1_minus_y0",
        "target_labels_excluded_from_X": True,
        "source_sha256": hashlib.sha256(read_sample_bytes(dataset, sample, data_dir)).hexdigest(),
        "train_rows_sha256": hashlib.sha256(train_idx.astype("<i8").tobytes()).hexdigest(),
        "test_rows_sha256": hashlib.sha256(test_idx.astype("<i8").tobytes()).hexdigest(),
    }
    return {
        "X_train": X_train, "A_train": A[train_idx], "Y_train": Y_train,
        "X_test": X_test, "tau_test": tau_test, "metadata": metadata,
        "train_indices": train_idx, "test_indices": test_idx,
    }


def run_one_dataset_sample(
    dataset: str,
    sample: int,
    data_dir=None,
    test_size: float = 0.5,
    seed: int = 50,
    scale_y_tau: bool = True,
    xi: float = 0.01,
    lams=None,
    kf_length: float = 1.3,
    kf_nu: float = 1.5,
    kh_length: float = 1.2,
    kh_nu: float = 2.5,
    krr_kwargs=None,
    max_train_size=None,
    max_test_size=None,
    verbose=True,
):
    """Run Ours, plug-in, DR and R with Nyström KRR.

    ``xi`` and ``lams`` are the diagonal penalties in K + alpha I, i.e.,
    alpha = n_fit * lambda for the normalized squared-loss convention.
    """
    if lams is None:
        lams = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    options = _resolved_krr_kwargs(krr_kwargs)
    prepared = prepare_realcause_sample(
        dataset, sample, data_dir, test_size, seed, scale_y_tau,
        max_train_size=max_train_size, max_test_size=max_test_size,
    )
    X_tr, A_tr, Y_tr = [prepared[k] for k in ["X_train", "A_train", "Y_train"]]
    X_te, tau_te = prepared["X_test"], prepared["tau_test"]
    result = dict(prepared["metadata"])
    if verbose:
        print(f"[Loaded] {dataset} sample={sample}: n={result['n_total']}, "
              f"d={result['d']}, train={len(X_tr)}, treated={A_tr.sum()}")
    KF = lambda a, b: matern_kernel(a, b, length_scale=kf_length, nu=kf_nu)
    KH = lambda a, b: matern_kernel(a, b, length_scale=kh_length, nu=kh_nu)
    start = perf_counter()
    predict, selected = fit_ours(
        X_tr, A_tr, Y_tr, xi, lams, KF, [("kh", KH)], seed,
        split_method="kfold", krr_kwargs=options,
    )
    mse_ours = float(np.mean((predict(X_te) - tau_te) ** 2))
    best_betas = [key[1] for key in selected]
    result.update(mse_ours=mse_ours, best_beta=best_betas,
                  seconds_ours=perf_counter() - start)
    start = perf_counter()
    h_plugin, best_lams = fit_plugin(
        X_full=X_tr, A_full=A_tr, Y_full=Y_tr,
        lambda_grid=lams, K_F_kernel_func=KF, random_state=seed,
        n_folds=3, krr_kwargs=options,
    )
    result.update(mse_plugin=float(np.mean((h_plugin(X_te) - tau_te) ** 2)),
                  best_lam0_plugin=best_lams[0], best_lam1_plugin=best_lams[1],
                  seconds_plugin=perf_counter() - start)
    for key, fit_fn in [("dr", fit_dr), ("r", fit_r_learner)]:
        start = perf_counter()
        predict = fit_fn(
            X_full=X_tr, A_full=A_tr, Y_full=Y_tr, random_state=seed,
            K_F_kernel_func=KF, K_H_kernel_func=KH, lambda_grid=lams,
            krr_kwargs=options,
        )
        result[f"mse_{key}"] = float(np.mean((predict(X_te) - tau_te) ** 2))
        result[f"seconds_{key}"] = perf_counter() - start
    result.update(
        backend="nystrom",
        nystrom_m=options["nystrom_m"],
        nystrom_random_state=options["nystrom_random_state"],
        xi_diagonal=float(xi), lambda_grid=list(lams),
        kf_length=float(kf_length), kf_nu=float(kf_nu),
        kh_length=float(kh_length), kh_nu=float(kh_nu),
    )
    result["seconds_total"] = sum(result[f"seconds_{k}"] for k in ["ours", "plugin", "dr", "r"])
    return result
