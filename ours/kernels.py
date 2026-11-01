"""Gaussian, Matérn, Sobolev kernels, and coordinate libraries."""

import math
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.optimize import brentq
from sklearn.gaussian_process.kernels import Matern, RBF


def gaussian_kernel(X1, X2, length_scale=1.0):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    sqdist = cdist(X1, X2, "sqeuclidean")
    return np.exp(-0.5 * sqdist / (float(length_scale) ** 2))


def matern_kernel(X1, X2, length_scale=1.0, nu=1.5):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    return Matern(length_scale=float(length_scale), nu=float(nu))(X1, X2)


def bernoulli_poly(n, x):
    x = np.asarray(x)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x - 0.5
    if n == 2:
        return x**2 - x + 1 / 6
    if n == 3:
        return x**3 - 1.5 * x**2 + 0.5 * x
    if n == 4:
        return x**4 - 2 * x**3 + x**2 - 1 / 30
    if n == 5:
        return x**5 - 2.5 * x**4 + (5 / 3) * x**3 - (1 / 6) * x
    if n == 6:
        return x**6 - 3 * x**5 + 2.5 * x**4 - 0.5 * x**2 + 1 / 42
    raise NotImplementedError(f"Bernoulli polynomial B_{n} not implemented.")


def sobolev_unanchored_kernel(X1, X2, alpha=1, gamma=1.0):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    x = X1[:, 0]
    y = X2[:, 0]
    K = np.ones((x.size, y.size))
    for tau in range(1, alpha + 1):
        Bt_x = bernoulli_poly(tau, x) / math.factorial(tau)
        Bt_y = bernoulli_poly(tau, y) / math.factorial(tau)
        K += gamma * np.outer(Bt_x, Bt_y)
    absdiff = np.abs(x[:, None] - y[None, :])
    K += -(((-1) ** alpha) * gamma * bernoulli_poly(2 * alpha, absdiff) / math.factorial(2 * alpha))
    return K


def _project_coords(X, coord_idx=None):
    X = np.atleast_2d(X)
    if coord_idx is None:
        return X
    return X[:, coord_idx]


def make_KH_candidates_multidim(
    d,
    length_scale_H=1.0,
    nu_grid=(2.5, 3.5),
    subset_modes=("all", "x1-4"),
    include_rbf=True,
):
    nu_grid = list(nu_grid)
    first4_idx = list(range(min(4, d)))
    x5_8_idx = list(range(4, min(8, d)))
    subset_map = {"all": None, "x1-4": first4_idx, "x5-8": x5_8_idx}
    candidates = []

    def get_ls(s_mode, val):
        if isinstance(length_scale_H, dict):
            return length_scale_H.get((s_mode, val), 1.0)
        return float(length_scale_H)

    for subset_name in subset_modes:
        if subset_name not in subset_map:
            continue
        idx = subset_map[subset_name]
        if idx == []:
            continue
        for nu in nu_grid:
            ls_val = get_ls(subset_name, nu)
            kname = f"matern_{subset_name}_nu{nu}_ls{ls_val}"
            K = (
                lambda X1, X2, idx=idx, nu=nu, ls=ls_val: matern_kernel(
                    _project_coords(X1, idx),
                    _project_coords(X2, idx),
                    length_scale=ls,
                    nu=nu,
                )
            )
            candidates.append((kname, K))

        if include_rbf:
            ls_val = get_ls(subset_name, "rbf")
            kname = f"rbf_{subset_name}_ls{ls_val}"
            K = (
                lambda X1, X2, idx=idx, ls=ls_val: gaussian_kernel(
                    _project_coords(X1, idx),
                    _project_coords(X2, idx),
                    length_scale=ls,
                )
            )
            candidates.append((kname, K))

    return candidates


def get_median_based_length_scales(
    d,
    subset_modes=("all", "x1-4"),
    nu_candidates=(1.5, 2.5),
    include_rbf=True,
    n_sample=2000,
    random_seed=42,
):
    """
    Compute length scales so that the kernel value at the median pairwise
    distance is 0.5. Returns a dict suitable for length_scale_H in
    make_KH_candidates_multidim.
    """
    rng = np.random.RandomState(random_seed)

    X_sample = rng.uniform(-1, 1, size=(n_sample, d))
    ls_dict = {}

    def get_kernel_val(dist, ls, nu):
        X_a = np.zeros((1, 1))
        X_b = np.array([[dist]])
        if nu == "rbf":
            k = RBF(length_scale=ls)
        else:
            k = Matern(length_scale=ls, nu=nu)
        return k(X_a, X_b)[0, 0]

    def objective(ls, median_dist, nu):
        return get_kernel_val(median_dist, ls, nu) - 0.5

    for mode in subset_modes:
        if mode == "all":
            X_sub = X_sample
        elif mode == "x1-4":
            X_sub = X_sample[:, :min(4, d)]
        elif mode == "x5-8":
            X_sub = X_sample[:, 4:min(8, d)]
            if X_sub.shape[1] == 0:
                raise ValueError("subset mode x5-8 requires d >= 5.")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        dists = pdist(X_sub, metric="euclidean")
        median_dist = np.median(dists)

        for nu in nu_candidates:
            optimal_ls = brentq(objective, 0.01, 100.0, args=(median_dist, nu))
            ls_dict[(mode, nu)] = round(optimal_ls, 4)

        if include_rbf:
            optimal_ls = brentq(objective, 0.01, 100.0, args=(median_dist, "rbf"))
            ls_dict[(mode, "rbf")] = round(optimal_ls, 4)

    return ls_dict
