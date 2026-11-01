"""Two-fold R-learner with weighted KRR for the residual loss."""

import numpy as np
from sklearn.model_selection import train_test_split
from ours.model_selection import fit_krr_kfold_cv, fit_weighted_krr_kfold_cv


def build_r_learner_pseudo_outcome(Y_residual, A_residual, min_abs_residual=1e-8):
    Y_residual = np.asarray(Y_residual, float).ravel()
    A_residual = np.asarray(A_residual, float).ravel()
    pseudo_y = np.divide(
        Y_residual,
        A_residual,
        out=np.zeros_like(Y_residual, dtype=float),
        where=np.abs(A_residual) > float(min_abs_residual),
    )
    sample_weight = A_residual**2
    return pseudo_y, sample_weight


def fit_r_learner(
    X_full,
    A_full,
    Y_full,
    random_state,
    K_F_kernel_func,
    K_H_kernel_func,
    lambda_grid,
    n_folds: int = 3,
    pi_clip=(0.01, 0.99),
    min_abs_residual: float = 1e-8,
    krr_kwargs=None,
):
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

    def _fit_r_once(X_nui, A_nui, Y_nui, X_stage, A_stage, Y_stage, seed_base: int):
        X_stage = np.asarray(X_stage)
        if X_stage.shape[0] == 0:
            return lambda x: np.zeros(np.atleast_2d(x).shape[0])

        m_hat, _ = fit_krr_kfold_cv(
            X_train=np.asarray(X_nui),
            y_train=np.asarray(Y_nui),
            kernel_func=K_F_kernel_func,
            lambda_candidates=lambda_grid,
            random_state=random_state,
            seed_offset=seed_base + 0,
            n_folds=n_folds,
            krr_kwargs=krr_kwargs,
        )
        pi_hat, _ = fit_krr_kfold_cv(
            X_train=np.asarray(X_nui),
            y_train=np.asarray(A_nui).astype(float),
            kernel_func=K_F_kernel_func,
            lambda_candidates=lambda_grid,
            random_state=random_state,
            seed_offset=seed_base + 1000,
            n_folds=n_folds,
            krr_kwargs=krr_kwargs,
        )

        pi_low, pi_high = pi_clip
        pi = np.clip(pi_hat(X_stage), float(pi_low), float(pi_high))
        y_residual = np.asarray(Y_stage, float).ravel() - m_hat(X_stage)
        a_residual = np.asarray(A_stage).astype(float).ravel() - pi
        pseudo_y, sample_weight = build_r_learner_pseudo_outcome(
            Y_residual=y_residual,
            A_residual=a_residual,
            min_abs_residual=min_abs_residual,
        )

        h_hat, _ = fit_weighted_krr_kfold_cv(
            X_train=X_stage,
            y_train=pseudo_y,
            sample_weight=sample_weight,
            kernel_func=K_H_kernel_func,
            lambda_candidates=lambda_grid,
            random_state=random_state,
            seed_offset=seed_base + 2000,
            n_folds=n_folds,
            krr_kwargs=krr_kwargs,
        )
        return h_hat

    h_12 = _fit_r_once(X_D1, A_D1, Y_D1, X_D2, A_D2, Y_D2, seed_base=0)
    h_21 = _fit_r_once(X_D2, A_D2, Y_D2, X_D1, A_D1, Y_D1, seed_base=5000)

    def h_cf(x):
        x = np.asarray(x)
        return 0.5 * (h_12(x) + h_21(x))

    return h_cf
