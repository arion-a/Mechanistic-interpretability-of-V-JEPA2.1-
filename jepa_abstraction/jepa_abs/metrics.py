"""ARS / gap metrics with paired group bootstrap (protocol 01, section 6)."""
import numpy as np

EPS = 0.0025


def ars(L_S, L_J, L_O):
    G = L_S - L_O
    return (L_S - L_J) / G if G != 0 else np.nan


def evaluate(loss_S, loss_J, loss_O, n_boot=2000, seed=0, eps=EPS):
    """Per-group losses (equal-weighted over laws already). Ratio of means; groups resampled jointly."""
    loss_S, loss_J, loss_O = map(np.asarray, (loss_S, loss_J, loss_O))
    n = len(loss_S)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n))
    S, J, O = (a[idx].mean(1) for a in (loss_S, loss_J, loss_O))
    G, B = S - O, S - J
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.where(G > 0, B / G, np.nan)
    pt = dict(L_S=loss_S.mean(), L_J=loss_J.mean(), L_O=loss_O.mean())
    pt["G"], pt["B"] = pt["L_S"] - pt["L_O"], pt["L_S"] - pt["L_J"]
    lo = lambda x: float(np.percentile(x, 2.5)); hi = lambda x: float(np.percentile(x, 97.5))
    pt.update(G_ci=(lo(G), hi(G)), B_ci=(lo(B), hi(B)))
    eligible = pt["G"] > eps and pt["G_ci"][0] > eps
    pt["status"] = "VALID" if eligible else "INVALID_GAP"
    if eligible:
        pt["ARS"] = ars(pt["L_S"], pt["L_J"], pt["L_O"])
        pt["ARS_ci"] = (lo(A), hi(A))
        pt["unstable_frac"] = float(np.isnan(A).mean())
    else:
        pt["ARS"] = None
    return pt
