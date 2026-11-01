"""Dense and Nyström ridge solvers, including weighted squared loss.

reg_param is the raw Gram diagonal penalty; normalized ridge is reg_param / n.
"""

import numpy as np


def _make_nystrom_feature_state(
    X_train,
    kernel_func,
    nystrom_m,
    random_state=None,
    nystrom_jitter: float = 1e-10,
    nystrom_rcond: float = 1e-12,
):
    X_train = np.atleast_2d(np.asarray(X_train, float))
    n = X_train.shape[0]
    m = min(int(nystrom_m), n)
    if m <= 0:
        raise ValueError("nystrom_m must be positive when provided.")

    seed = 0 if random_state is None else random_state
    rng = np.random.default_rng(seed)
    if m == n:
        landmark_idx = np.arange(n)
    else:
        landmark_idx = np.sort(rng.choice(n, size=m, replace=False))

    X_landmarks = X_train[landmark_idx]
    W = kernel_func(X_landmarks, X_landmarks)
    W = 0.5 * (W + W.T)
    if nystrom_jitter > 0:
        W = W + float(nystrom_jitter) * np.eye(m)

    eigvals, eigvecs = np.linalg.eigh(W)
    max_eval = float(np.max(eigvals)) if eigvals.size else 0.0
    cutoff = max(float(nystrom_rcond) * max(max_eval, 1.0), 0.0)
    keep = eigvals > cutoff
    if not np.any(keep):
        keep = eigvals > 0
    if not np.any(keep):
        raise np.linalg.LinAlgError("Nystrom landmark kernel has no positive eigenvalues.")

    transform = eigvecs[:, keep] / np.sqrt(eigvals[keep])[None, :]
    Phi_train = kernel_func(X_train, X_landmarks) @ transform
    return {
        "landmark_idx": landmark_idx,
        "X_landmarks": X_landmarks,
        "transform": transform,
        "Phi_train": Phi_train,
    }


def _nystrom_features(X, kernel_func, X_landmarks, transform):
    X = np.atleast_2d(np.asarray(X, float))
    return kernel_func(X, X_landmarks) @ transform


class KRR:
    """
    Kernel ridge regressor with configurable numerical stabilization.

    Parameters:
    - reg_param: raw Gram-matrix ridge (normalized-loss penalty = reg_param / n)
    - jitter: added as reg_param + jitter on diagonal before solve
    - lstsq_fallback: if True, LinAlgError falls back to lstsq
    - fallback_jitter: extra diagonal jitter used in fallback branch
    - nystrom_m: if provided, use min(n, nystrom_m) Nyström landmarks
    """

    def __init__(
        self,
        kernel_func,
        reg_param,
        jitter: float = 0.0,
        lstsq_fallback: bool = False,
        fallback_jitter: float = 1e-6,
        nystrom_m=None,
        nystrom_random_state=None,
        nystrom_jitter: float = 1e-10,
        nystrom_rcond: float = 1e-12,
    ):
        self.kernel_func = kernel_func
        self.reg_param = float(reg_param)
        self.jitter = float(jitter)
        self.lstsq_fallback = bool(lstsq_fallback)
        self.fallback_jitter = float(fallback_jitter)
        self.nystrom_m = None if nystrom_m is None else int(nystrom_m)
        self.nystrom_random_state = nystrom_random_state
        self.nystrom_jitter = float(nystrom_jitter)
        self.nystrom_rcond = float(nystrom_rcond)
        self.X_train = None
        self.alpha = None
        self.nystrom_coef = None
        self.nystrom_landmarks = None
        self.nystrom_transform = None
        self.nystrom_landmark_idx = None
        self._use_nystrom = False

    def _should_use_nystrom(self, n):
        return self.nystrom_m is not None and self.nystrom_m > 0

    def fit(self, X, y):
        self.X_train = np.atleast_2d(X)
        y = np.asarray(y, float).ravel()
        n = self.X_train.shape[0]
        self.alpha = None
        self.nystrom_coef = None
        self.nystrom_landmarks = None
        self.nystrom_transform = None
        self.nystrom_landmark_idx = None
        self._use_nystrom = False
        if n == 0:
            self.alpha = np.array([])
            return
        if self._should_use_nystrom(n):
            state = _make_nystrom_feature_state(
                X_train=self.X_train,
                kernel_func=self.kernel_func,
                nystrom_m=self.nystrom_m,
                random_state=self.nystrom_random_state,
                nystrom_jitter=self.nystrom_jitter,
                nystrom_rcond=self.nystrom_rcond,
            )
            Phi = state["Phi_train"]
            gram = Phi.T @ Phi
            rhs = Phi.T @ y
            self.nystrom_coef = self._solve(gram, rhs)
            self.nystrom_landmarks = state["X_landmarks"]
            self.nystrom_transform = state["transform"]
            self.nystrom_landmark_idx = state["landmark_idx"]
            self._use_nystrom = True
            return

        K = self.kernel_func(self.X_train, self.X_train)
        self.alpha = self._solve(K, y)

    def _solve(self, gram, rhs):
        eye = np.eye(gram.shape[0])
        try:
            return np.linalg.solve(gram + (self.reg_param + self.jitter) * eye, rhs)
        except np.linalg.LinAlgError:
            ridge = self.reg_param + max(10.0 * self.jitter, self.fallback_jitter)
            if self.lstsq_fallback:
                return np.linalg.lstsq(gram + ridge * eye, rhs, rcond=None)[0]
            return np.linalg.solve(gram + ridge * eye, rhs)

    def predict(self, X_test):
        X_test = np.atleast_2d(X_test)
        if self._use_nystrom:
            if (
                self.nystrom_coef is None
                or self.nystrom_landmarks is None
                or self.nystrom_transform is None
            ):
                return np.zeros(X_test.shape[0])
            Phi_test = _nystrom_features(
                X_test,
                self.kernel_func,
                self.nystrom_landmarks,
                self.nystrom_transform,
            )
            return Phi_test @ self.nystrom_coef
        if (
            self.X_train is None
            or self.X_train.shape[0] == 0
            or self.alpha is None
            or self.alpha.size == 0
        ):
            return np.zeros(X_test.shape[0])
        return self.kernel_func(X_test, self.X_train) @ self.alpha


class WeightedKRR(KRR):
    """KRR for sum_i w_i (y_i - f(x_i))^2 + reg_param ||f||^2."""

    def fit(self, X, y, sample_weight):
        self.X_train = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, float).ravel()
        w = np.maximum(np.asarray(sample_weight, float).ravel(), 0.0)
        n = self.X_train.shape[0]
        self.alpha = None
        self.nystrom_coef = None
        self.nystrom_landmarks = None
        self.nystrom_transform = None
        self.nystrom_landmark_idx = None
        self._use_nystrom = False
        if n == 0 or np.sum(w) <= 0:
            self.alpha = np.zeros(n)
            return

        if self._should_use_nystrom(n):
            state = _make_nystrom_feature_state(
                X_train=self.X_train,
                kernel_func=self.kernel_func,
                nystrom_m=self.nystrom_m,
                random_state=self.nystrom_random_state,
                nystrom_jitter=self.nystrom_jitter,
                nystrom_rcond=self.nystrom_rcond,
            )
            Phi = state["Phi_train"]
            sqrt_w = np.sqrt(w)
            Phi_w = sqrt_w[:, None] * Phi
            gram = Phi_w.T @ Phi_w
            rhs = Phi.T @ (w * y)
            self.nystrom_coef = self._solve(gram, rhs)
            self.nystrom_landmarks = state["X_landmarks"]
            self.nystrom_transform = state["transform"]
            self.nystrom_landmark_idx = state["landmark_idx"]
            self._use_nystrom = True
            return

        sqrt_w = np.sqrt(w)
        K = self.kernel_func(self.X_train, self.X_train)
        K_tilde = (sqrt_w[:, None] * K) * sqrt_w[None, :]
        rhs = sqrt_w * y
        gamma = self._solve(K_tilde, rhs)
        self.alpha = sqrt_w * gamma

    def predict(self, X_test):
        return super().predict(np.asarray(X_test, float))
