"""Switch-imputation estimator with kernel selection and cyclic averaging."""

import numpy as np
from sklearn.model_selection import KFold, train_test_split
from ours.krr import KRR


def fit_ours(
    X, A, Y, xi, lambda_grid, K_F, K_H_candidates, random_state,
    *, crossfit=True, split_method="shuffle", krr_kwargs=None,
):
    """Return a predictor and the selected (kernel, ridge) pairs.

    Synthetic uses shuffled three-way splits; RealCause uses KFold partitions.
    Each rotation fits on D1, learns validation nuisances on D2, and selects
    on D3. Test outcomes are never passed to the estimator.
    """
    X = np.asarray(X)
    A = np.asarray(A).astype(int).ravel()
    Y = np.asarray(Y, float).ravel()
    lambda_grid = list(lambda_grid)
    if len(X) < 3 or not lambda_grid or not K_H_candidates:
        raise ValueError("Need at least three rows, a ridge grid, and a target kernel.")
    if split_method == "shuffle":
        parts = split_three_way_data(X, A, Y, X.shape[1], random_state)
        folds = [(parts[i], parts[i + 3], parts[i + 6]) for i in range(3)]
    elif split_method == "kfold":
        splitter = KFold(n_splits=3, shuffle=True, random_state=random_state)
        folds = [(X[idx], A[idx], Y[idx]) for _, idx in splitter.split(X)]
    else:
        raise ValueError("split_method must be 'shuffle' or 'kfold'.")

    models, selected = [], []
    for rotation in range(3 if crossfit else 1):
        train = folds[rotation]
        nuisance = folds[(rotation + 1) % 3]
        validation = folds[(rotation + 2) % 3]
        candidates = fit_switch_candidates(
            *train, xi, lambda_grid, K_F, K_H_candidates, krr_kwargs=krr_kwargs,
        )
        key = select_switch_candidate_key(
            candidates, *nuisance, *validation, xi, K_F, krr_kwargs=krr_kwargs,
        )
        models.append(candidates[key])
        selected.append(key)

    def predict(x):
        return np.mean([model.predict(np.asarray(x)) for model in models], axis=0)

    return predict, selected


def build_switch_pseudo_outcome(A, Y, mu0_hat, mu1_hat):
    A = np.asarray(A).astype(int).ravel()
    Y = np.asarray(Y, float).ravel()
    mu0_hat = np.asarray(mu0_hat, float).ravel()
    mu1_hat = np.asarray(mu1_hat, float).ravel()
    return np.where(A == 1, Y - mu0_hat, mu1_hat - Y)


def split_three_way_data(X_full, A_full, Y_full, d, random_state_seed):
    X_d1, X_tmp, A_d1, A_tmp, Y_d1, Y_tmp = train_test_split(
        X_full,
        A_full,
        Y_full,
        test_size=2 / 3,
        random_state=random_state_seed,
        shuffle=True,
        stratify=None,
    )

    if X_tmp.shape[0] < 2:
        X_d2 = X_tmp
        X_d3 = np.empty((0, d))
        A_d2 = A_tmp
        A_d3 = np.array([])
        Y_d2 = Y_tmp
        Y_d3 = np.array([])
    else:
        X_d2, X_d3, A_d2, A_d3, Y_d2, Y_d3 = train_test_split(
            X_tmp,
            A_tmp,
            Y_tmp,
            test_size=0.5,
            random_state=random_state_seed + 1,
            shuffle=True,
            stratify=None,
        )

    return X_d1, X_d2, X_d3, A_d1, A_d2, A_d3, Y_d1, Y_d2, Y_d3


def fit_switch_candidates(
    X_d1,
    A_d1,
    Y_d1,
    xi,
    lambda_grid,
    K_F_kernel_func,
    K_H_candidates,
    krr_kwargs=None,
):
    lambda_grid = [float(v) for v in lambda_grid]
    if len(lambda_grid) == 0:
        raise ValueError("lambda_grid must contain at least one value.")
    if not K_H_candidates:
        return {}
    if krr_kwargs is None:
        krr_kwargs = {}

    f0_d1 = KRR(K_F_kernel_func, xi, **krr_kwargs)
    f0_d1.fit(X_d1[np.asarray(A_d1).astype(int) == 0], Y_d1[np.asarray(A_d1).astype(int) == 0])
    f1_d1 = KRR(K_F_kernel_func, xi, **krr_kwargs)
    f1_d1.fit(X_d1[np.asarray(A_d1).astype(int) == 1], Y_d1[np.asarray(A_d1).astype(int) == 1])

    m_d1 = build_switch_pseudo_outcome(
        A=A_d1,
        Y=Y_d1,
        mu0_hat=f0_d1.predict(X_d1),
        mu1_hat=f1_d1.predict(X_d1),
    )

    candidate_models = {}
    for kernel_name, K_H in K_H_candidates:
        for lam in lambda_grid:
            h_model = KRR(K_H, lam, **krr_kwargs)
            h_model.fit(X_d1, m_d1)
            candidate_models[(kernel_name, float(lam))] = h_model
    return candidate_models


def select_switch_candidate_key(
    candidate_models,
    X_d2,
    A_d2,
    Y_d2,
    X_d3,
    A_d3,
    Y_d3,
    xi,
    K_F_kernel_func,
    krr_kwargs=None,
):
    keys = list(candidate_models.keys())
    if not keys:
        return None
    if (np.asarray(X_d2).shape[0] == 0) or (np.asarray(X_d3).shape[0] == 0):
        return keys[0]

    if krr_kwargs is None:
        krr_kwargs = {}

    A_d2_int = np.asarray(A_d2).astype(int)
    f0_d2 = KRR(K_F_kernel_func, xi, **krr_kwargs)
    f0_d2.fit(X_d2[A_d2_int == 0], np.asarray(Y_d2)[A_d2_int == 0])
    f1_d2 = KRR(K_F_kernel_func, xi, **krr_kwargs)
    f1_d2.fit(X_d2[A_d2_int == 1], np.asarray(Y_d2)[A_d2_int == 1])

    m_tilde_d3 = build_switch_pseudo_outcome(
        A=A_d3,
        Y=Y_d3,
        mu0_hat=f0_d2.predict(X_d3),
        mu1_hat=f1_d2.predict(X_d3),
    )

    best_key = None
    min_loss = float("inf")
    for key, h_model in candidate_models.items():
        loss = float(np.mean((m_tilde_d3 - h_model.predict(X_d3)) ** 2))
        if loss < min_loss:
            min_loss = loss
            best_key = key
    return keys[0] if best_key is None else best_key
