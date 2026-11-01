"""Two-fold DR and Oracle DR learners."""

import numpy as np
from sklearn.model_selection import train_test_split
from ours.model_selection import fit_krr_kfold_cv


def build_dr_pseudo_outcome(A, Y, mu0_hat, mu1_hat, pi_hat, pi_clip=(0.01, 0.99)):
    A = np.asarray(A, float).ravel()
    Y = np.asarray(Y, float).ravel()
    mu0_hat = np.asarray(mu0_hat, float).ravel()
    mu1_hat = np.asarray(mu1_hat, float).ravel()
    pi_low, pi_high = pi_clip
    pi_hat = np.clip(np.asarray(pi_hat, float).ravel(), float(pi_low), float(pi_high))
    return (
        (mu1_hat - mu0_hat)
        + (A / pi_hat) * (Y - mu1_hat)
        - ((1.0 - A) / (1.0 - pi_hat)) * (Y - mu0_hat)
    )


def fit_dr(
    X_full,
    A_full,
    Y_full,
    random_state,
    K_F_kernel_func,
    K_H_kernel_func,
    lambda_grid,
    n_folds: int = 3,
    pi_clip=(0.01, 0.99),
    krr_kwargs=None,
    pi_func=None,
):
    """Use a known propensity (pi_func) for Oracle DR; otherwise estimate it."""
    X_full = np.asarray(X_full, float)
    A_full = np.asarray(A_full).astype(int).ravel()
    Y_full = np.asarray(Y_full, float).ravel()

    lambda_grid = [float(v) for v in lambda_grid]
    if len(lambda_grid) == 0:
        raise ValueError("lambda_grid must contain at least one value.")

    X_D1, X_D2, A_D1, A_D2, Y_D1, Y_D2 = train_test_split(
        X_full,
        A_full,
        Y_full,
        test_size=0.5,
        random_state=random_state,
        shuffle=True,
        stratify=None,
    )

    def _fit_dr_once(X_nui, A_nui, Y_nui, X_stage, A_stage, Y_stage, seed_base: int):
        X_stage = np.asarray(X_stage)
        if X_stage.shape[0] == 0:
            return lambda x: np.zeros(np.atleast_2d(x).shape[0])

        A_nui = np.asarray(A_nui).astype(int).ravel()
        f0_hat, _ = fit_krr_kfold_cv(
            X_train=np.asarray(X_nui)[A_nui == 0],
            y_train=np.asarray(Y_nui)[A_nui == 0],
            kernel_func=K_F_kernel_func,
            lambda_candidates=lambda_grid,
            random_state=random_state,
            seed_offset=seed_base + 0,
            n_folds=n_folds,
            krr_kwargs=krr_kwargs,
        )
        f1_hat, _ = fit_krr_kfold_cv(
            X_train=np.asarray(X_nui)[A_nui == 1],
            y_train=np.asarray(Y_nui)[A_nui == 1],
            kernel_func=K_F_kernel_func,
            lambda_candidates=lambda_grid,
            random_state=random_state,
            seed_offset=seed_base + 1000,
            n_folds=n_folds,
            krr_kwargs=krr_kwargs,
        )
        if pi_func is None:
            pi_hat, _ = fit_krr_kfold_cv(
                X_train=np.asarray(X_nui),
                y_train=np.asarray(A_nui).astype(float),
                kernel_func=K_F_kernel_func,
                lambda_candidates=lambda_grid,
                random_state=random_state,
                seed_offset=seed_base + 2000,
                n_folds=n_folds,
                krr_kwargs=krr_kwargs,
            )

        mu0 = f0_hat(X_stage)
        mu1 = f1_hat(X_stage)
        pi = pi_hat(X_stage) if pi_func is None else pi_func(X_stage)
        m = build_dr_pseudo_outcome(
            A=A_stage,
            Y=Y_stage,
            mu0_hat=mu0,
            mu1_hat=mu1,
            pi_hat=pi,
            pi_clip=pi_clip,
        )

        h_hat, _ = fit_krr_kfold_cv(
            X_train=X_stage,
            y_train=m,
            kernel_func=K_H_kernel_func,
            lambda_candidates=lambda_grid,
            random_state=random_state,
            seed_offset=seed_base + 3000,
            n_folds=n_folds,
            krr_kwargs=krr_kwargs,
        )
        return h_hat

    h_12 = _fit_dr_once(X_D1, A_D1, Y_D1, X_D2, A_D2, Y_D2, seed_base=0)
    h_21 = _fit_dr_once(X_D2, A_D2, Y_D2, X_D1, A_D1, Y_D1, seed_base=5000)

    def h_cf(x):
        x = np.asarray(x)
        return 0.5 * (h_12(x) + h_21(x))

    return h_cf
