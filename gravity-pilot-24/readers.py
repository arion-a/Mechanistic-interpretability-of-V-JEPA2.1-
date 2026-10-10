"""Pure-numpy reader/metric library for the 24-video gravity pilot (protocol v1.0 sections 5-9). No model, no I/O."""
import math
import numpy as np

LAMBDA = 0.01
S_MIN = 1e-12
RHO_MAX = 1.10
E_FULL_MIN = 1e-9


class Degenerate(Exception):
    pass


def fit_scaler(R_train):
    """Training-only mean and single scalar RMS scale (section 5)."""
    R = np.asarray(R_train, np.float64)
    mean = R.mean(0)
    s = math.sqrt(float(((R - mean) ** 2).sum(1).mean()))
    if s <= S_MIN:
        raise Degenerate(f"scale {s:.3e} <= {S_MIN}")
    return mean, s


def ridge_dual(A, Y, lam=LAMBDA):
    """W = A^T (A A^T + n*lam*I)^-1 (Y - Ybar); linear solve, no explicit inverse; intercept Ybar is unpenalised."""
    A, Y = np.asarray(A, np.float64), np.asarray(Y, np.float64)
    n = A.shape[0]
    Ybar = Y.mean(0)
    alpha = np.linalg.solve(A @ A.T + n * lam * np.eye(n), Y - Ybar)
    return A.T @ alpha, Ybar


def fit_predict(R_train, Y_train, R_test):
    mean, s = fit_scaler(R_train)
    A = (np.asarray(R_train, np.float64) - mean) / s
    W, Ybar = ridge_dual(A, Y_train)
    At = (np.asarray(R_test, np.float64) - mean) / s
    return At @ W + Ybar, dict(mean=mean, scale=s, W=W, Ybar=Ybar)


def color_label(score):
    return (np.asarray(score) > 0).astype(int)  # 1 = blue; a score of exactly 0 -> orange


def euclid_rmse(err):
    err = np.asarray(err, float)
    return float(np.sqrt(np.mean(err ** 2)))


def rho(E_proj, E_full):
    if E_full <= E_FULL_MIN:
        return None, f"E_full={E_full:.3e} <= {E_FULL_MIN}: rho undefined; report absolute change"
    return E_proj / E_full, None
