"""Cross-validation and refitting for unweighted and weighted KRR."""

import numpy as np
from sklearn.model_selection import KFold
from ours.krr import KRR, WeightedKRR


def fit_krr_kfold_cv(
    X_train,
    y_train,
    kernel_func,
    lambda_candidates,
    random_state,
    seed_offset=0,
    n_folds: int = 3,
    krr_kwargs=None,
):
    X_train = np.asarray(X_train, float)
    y_train = np.asarray(y_train, float).ravel()
    lambda_candidates = [float(v) for v in lambda_candidates]
    if len(lambda_candidates) == 0:
        raise ValueError("lambda_candidates must contain at least one value.")

    if krr_kwargs is None:
        krr_kwargs = {}

    n = X_train.shape[0]
    if n == 0:
        return (lambda x_eval: np.zeros(np.atleast_2d(x_eval).shape[0])), float(lambda_candidates[0])
    if n == 1:
        const_val = float(y_train[0])
        return (lambda x_eval: np.full(np.atleast_2d(x_eval).shape[0], const_val)), float(lambda_candidates[0])

    k_eff = min(max(2, int(n_folds)), n)
    seed = None if random_state is None else int(random_state) + int(seed_offset)
    kf = KFold(n_splits=k_eff, shuffle=True, random_state=seed)
    splits = list(kf.split(X_train))

    best_lambda = float(lambda_candidates[0])
    best_cv_mse = np.inf

    for lam in lambda_candidates:
        total_sq_error = 0.0
        total_count = 0
        for train_idx, val_idx in splits:
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            model = KRR(kernel_func=kernel_func, reg_param=float(lam), **krr_kwargs)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            total_sq_error += float(np.sum((y_val - y_pred) ** 2))
            total_count += int(y_val.shape[0])
        cv_mse = np.inf if total_count == 0 else (total_sq_error / total_count)
        if cv_mse < best_cv_mse:
            best_cv_mse = cv_mse
            best_lambda = float(lam)

    final_model = KRR(kernel_func=kernel_func, reg_param=best_lambda, **krr_kwargs)
    final_model.fit(X_train, y_train)

    def f_hat(x_eval):
        X_eval = np.atleast_2d(np.asarray(x_eval, float))
        return final_model.predict(X_eval)

    return f_hat, best_lambda


def fit_weighted_krr_kfold_cv(
    X_train,
    y_train,
    sample_weight,
    kernel_func,
    lambda_candidates,
    random_state,
    seed_offset=0,
    n_folds: int = 3,
    krr_kwargs=None,
):
    X_train = np.asarray(X_train, float)
    y_train = np.asarray(y_train, float).ravel()
    sample_weight = np.maximum(np.asarray(sample_weight, float).ravel(), 0.0)
    lambda_candidates = [float(v) for v in lambda_candidates]
    if len(lambda_candidates) == 0:
        raise ValueError("lambda_candidates must contain at least one value.")

    if krr_kwargs is None:
        krr_kwargs = {}

    n = X_train.shape[0]
    if n == 0:
        return (lambda x_eval: np.zeros(np.atleast_2d(x_eval).shape[0])), float(lambda_candidates[0])
    if n == 1 or np.sum(sample_weight) <= 0:
        const_val = float(y_train[0])
        return (lambda x_eval: np.full(np.atleast_2d(x_eval).shape[0], const_val)), float(lambda_candidates[0])

    k_eff = min(max(2, int(n_folds)), n)
    seed = None if random_state is None else int(random_state) + int(seed_offset)
    kf = KFold(n_splits=k_eff, shuffle=True, random_state=seed)
    splits = list(kf.split(X_train))

    best_lambda = float(lambda_candidates[0])
    best_cv_loss = np.inf

    for lam in lambda_candidates:
        total_loss = 0.0
        total_weight = 0.0
        total_count = 0
        for train_idx, val_idx in splits:
            model = WeightedKRR(kernel_func=kernel_func, reg_param=float(lam), **krr_kwargs)
            model.fit(X_train[train_idx], y_train[train_idx], sample_weight[train_idx])
            y_pred = model.predict(X_train[val_idx])
            w_val = sample_weight[val_idx]
            sq_error = (y_train[val_idx] - y_pred) ** 2
            if np.sum(w_val) > 0:
                total_loss += float(np.sum(w_val * sq_error))
                total_weight += float(np.sum(w_val))
            else:
                total_loss += float(np.sum(sq_error))
                total_count += int(sq_error.shape[0])
        denom = total_weight if total_weight > 0 else max(total_count, 1)
        cv_loss = total_loss / denom
        if cv_loss < best_cv_loss:
            best_cv_loss = cv_loss
            best_lambda = float(lam)

    final_model = WeightedKRR(kernel_func=kernel_func, reg_param=best_lambda, **krr_kwargs)
    final_model.fit(X_train, y_train, sample_weight)

    def f_hat(x_eval):
        X_eval = np.atleast_2d(np.asarray(x_eval, float))
        return final_model.predict(X_eval)

    return f_hat, best_lambda
