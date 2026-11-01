"""Synthetic data-generating models and experiment kernel recipes."""

import numpy as np
from ours.kernels import (
    matern_kernel, sobolev_unanchored_kernel, _project_coords,
    make_KH_candidates_multidim,
)


def get_true_functions(d, p=None, case_type=""):
    if "Univariate" in case_type:
        def f0_star(x):
            return 5 * (np.abs(x[:, 0] - 0.4) + np.abs(x[:, 0] - 0.8))
        def h_star(x):
            return x[:, 0] * x[:, 0]
        def f1_star(x):
            return f0_star(x) + h_star(x)
    elif p is not None:
        if p > d:
            raise ValueError("p > d")
        f0_star = lambda x: 2 * np.sum(np.sin(np.atleast_2d(x)), axis=1) / d
        h_star = lambda x: 0.3 * np.sum(np.atleast_2d(x)[..., :p] ** 2, axis=-1) / p
        f1_star = lambda x: f0_star(x) + h_star(x)
    else:
        f0_star = lambda x: 2 * np.sum(np.sin(np.atleast_2d(x)), axis=1) / d
        h_star = lambda x: 0.5 * np.sum(np.atleast_2d(x), axis=1) / d
        f1_star = lambda x: f0_star(x) + h_star(x)
    return f0_star, f1_star, h_star


def make_kappa_propensity(kappa):
    if not (0 < float(kappa) <= 0.5):
        raise ValueError("kappa must lie in (0, 0.5].")

    def pi_func(x):
        val = np.sin(5 * np.linalg.norm(np.atleast_2d(x), axis=1))
        return np.clip(val, float(kappa), 1.0 - float(kappa))

    return pi_func


def generate_data(
    n_samples,
    d,
    f0_star,
    h_star,
    pi_func,
    noise_std,
    random_state=None,
    domain=(-1, 1),
):
    rng = np.random.RandomState(random_state)

    X = rng.uniform(domain[0], domain[1], size=(n_samples, d))
    pi_x = pi_func(X)
    A = rng.binomial(1, pi_x)

    mu0 = f0_star(X)
    h = h_star(X)
    mu1 = mu0 + h

    epsilon = rng.normal(0, noise_std, size=n_samples)
    Y_obs = (1 - A) * mu0 + A * mu1 + epsilon
    return X, A, Y_obs, mu0, mu1, h, pi_x


def calculate_mse(h_hat_func, h_star_func, X_test):
    pred = h_hat_func(X_test).ravel()
    true = h_star_func(X_test).ravel()
    return np.mean((pred - true) ** 2)


def make_synthetic_method_kernels(
    case_type,
    d,
    p_dim=None,
    length_scale_F=1.0,
    nu_F=1.5,
    length_scale_H=1.0,
    nu_H_candidates=(2.5, 3.5),
    subset_modes=("all", "x1-4"),
    include_rbf=True,
    nu_H_for_DR=2.5,
    univariate_F_alpha=1,
    univariate_H_alpha=2,
):
    is_univariate = ("Univariate" in case_type) or (d == 1)

    if is_univariate:
        K_F_ours = lambda X1, X2: sobolev_unanchored_kernel(
            X1, X2, alpha=univariate_F_alpha
        )
        K_H_candidates = [
            (
                f"sobolev_unanchored_a{univariate_H_alpha}",
                lambda X1, X2: sobolev_unanchored_kernel(
                    X1, X2, alpha=univariate_H_alpha
                ),
            ),
        ]
        K_H_for_DR = lambda X1, X2: sobolev_unanchored_kernel(
            X1, X2, alpha=univariate_H_alpha
        )
    else:
        K_F_ours = lambda X1, X2: matern_kernel(X1, X2, length_scale=length_scale_F, nu=nu_F)
        K_H_candidates = make_KH_candidates_multidim(
            d=d,
            length_scale_H=length_scale_H,
            nu_grid=nu_H_candidates,
            subset_modes=subset_modes,
            include_rbf=include_rbf,
        )

        is_subset_dr = (p_dim is not None) and (p_dim < d)
        subset_key_dr = "x1-4" if is_subset_dr else "all"
        if isinstance(length_scale_H, dict):
            ls_dr = length_scale_H.get((subset_key_dr, nu_H_for_DR), 1.0)
        else:
            ls_dr = float(length_scale_H)

        if is_subset_dr:
            idx = list(range(p_dim))
            K_H_for_DR = (
                lambda X1, X2, idx=idx, ls=ls_dr, nu_val=nu_H_for_DR: matern_kernel(
                    _project_coords(X1, idx), _project_coords(X2, idx), length_scale=ls, nu=nu_val
                )
            )
        else:
            K_H_for_DR = (
                lambda X1, X2, ls=ls_dr, nu_val=nu_H_for_DR: matern_kernel(
                    X1, X2, length_scale=ls, nu=nu_val
                )
            )

    return K_F_ours, K_H_candidates, K_H_for_DR, K_H_for_DR
