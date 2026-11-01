"""Plug-in estimator with separate cross-validated outcome regressions."""

import numpy as np
from ours.model_selection import fit_krr_kfold_cv


def fit_plugin(
    X_full,
    A_full,
    Y_full,
    lambda_grid,
    K_F_kernel_func,
    random_state,
    n_folds: int = 3,
    krr_kwargs=None,
):
    X_full = np.asarray(X_full, float)
    A_full = np.asarray(A_full).astype(int).ravel()
    Y_full = np.asarray(Y_full, float).ravel()

    lambda_grid = [float(v) for v in lambda_grid]
    if len(lambda_grid) == 0:
        raise ValueError("lambda_grid must contain at least one value.")

    X0, Y0 = X_full[A_full == 0], Y_full[A_full == 0]
    X1, Y1 = X_full[A_full == 1], Y_full[A_full == 1]

    f0_hat, best_lambda0 = fit_krr_kfold_cv(
        X_train=X0,
        y_train=Y0,
        kernel_func=K_F_kernel_func,
        lambda_candidates=lambda_grid,
        random_state=random_state,
        seed_offset=0,
        n_folds=n_folds,
        krr_kwargs=krr_kwargs,
    )
    f1_hat, best_lambda1 = fit_krr_kfold_cv(
        X_train=X1,
        y_train=Y1,
        kernel_func=K_F_kernel_func,
        lambda_candidates=lambda_grid,
        random_state=random_state,
        seed_offset=1000,
        n_folds=n_folds,
        krr_kwargs=krr_kwargs,
    )

    def h_plugin(x):
        x = np.asarray(x, float)
        return f1_hat(x) - f0_hat(x)

    return h_plugin, (float(best_lambda0), float(best_lambda1))
