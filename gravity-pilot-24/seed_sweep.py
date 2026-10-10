#!/usr/bin/env python3
"""Follow-up F1: projection-seed robustness (see protocol/FOLLOWUP_F1_SEED_SWEEP.md). No inference; reuses saved embeddings."""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import pilot as P, readers as RD   # noqa: E402

RES, OUT = ROOT / "results", ROOT / "results" / "followup_f1_seed_sweep"
SEEDS = [20261009] + list(range(20261010, 20261110))


def proj(seed):
    b = np.random.Generator(np.random.PCG64(seed)).integers(0, 2, size=(16, 768))
    return ((2 * b - 1) / 4).astype(np.float32)


def main():
    ok, why = P.receipt_ok()
    assert ok, why
    OUT.mkdir(exist_ok=True)
    man = json.loads((P.DS / "manifest.json").read_text()); clips = man["clips"]
    prov = json.loads((RES / "provenance.json").read_text()); S = json.loads((RES / "summary.json").read_text())
    assert prov["manifest_sha256"] == P.sha256_file(P.DS / "manifest.json")
    Z = np.load(RES / "embeddings_full.npy"); assert Z.shape == (24, 768)
    assert hashlib.sha256(Z.tobytes()).hexdigest() == prov["embeddings_full_sha256"]
    scene = np.array([c["scene"] for c in clips]); gi = np.array([c["gravity_idx"] for c in clips]); col = np.array([c["color_idx"] for c in clips])
    T = np.array([c["target_t4_m"] for c in clips], float)
    E_full, E_cv, E_prior = S["pooled_position_rmse_m"]["full"], S["pooled_position_rmse_m"]["cv"], S["pooled_position_rmse_m"]["prior"]
    bar = min(E_cv, E_prior)
    cvg = np.array([[0.0] * 2 for _ in clips]); gcv = {}
    for f in range(6):  # group baselines are constant (balanced groups) but computed, not assumed
        gcv[f] = (S["group_position_rmse_m"]["cv"][str(f)], S["group_position_rmse_m"]["prior"][str(f)])
    rows = []
    for seed in SEEDS:
        R = proj(seed); U = (Z @ R.T).astype(np.float32).astype(np.float64)
        pos = np.zeros((24, 2)); sc = np.zeros(24)
        for f in range(6):
            tr, te = scene != f, scene == f
            pos[te] = 100 * RD.fit_predict(U[tr], T[tr] / 100.0, U[te])[0]
            sc[te] = RD.fit_predict(U[tr], np.where(col[tr] == 1, 1.0, -1.0)[:, None], U[te])[0][:, 0]
        err = np.linalg.norm(pos - T, axis=1); E = RD.euclid_rmse(err)
        grp = [RD.euclid_rmse(err[scene == f]) for f in range(6)]
        beat = sum(g < min(gcv[f]) for f, g in enumerate(grp))
        corr = int(np.sum(RD.color_label(sc) == col))
        d = []
        for s in range(6):
            for a in range(2):
                i0 = int(np.where((scene == s) & (gi == a) & (col == 0))[0][0]); i1 = int(np.where((scene == s) & (gi == a) & (col == 1))[0][0])
                d.append(np.sum((pos[i1] - pos[i0]) ** 2))
        r = E / E_full
        rows.append(dict(seed=seed, projection_sha256=hashlib.sha256(R.tobytes()).hexdigest()[:16], E_projected_m=E, rho=r, groups_beating_both_baselines=beat,
                         color_correct_of_24=corr, J_projected_m=float(np.sqrt(np.mean(d))), below_bar=bool(E < bar), passes_primary=bool(E < bar and r <= RD.RHO_MAX),
                         **{f"E_group{f}_m": g for f, g in enumerate(grp)}))
    # validity check: the original seed must reproduce the pilot exactly
    r0 = rows[0]
    assert abs(r0["E_projected_m"] - S["pooled_position_rmse_m"]["projected"]) < 1e-9 and r0["color_correct_of_24"] == S["color"]["correct_of_24"]["projected"] \
        and abs(r0["J_projected_m"] - S["paired_color_sensitivity_J_m"]["projected"]) < 1e-9 and r0["projection_sha256"] == prov["projection_sha256"][:16], "seed 20261009 does not reproduce the pilot"
    with open(OUT / "seed_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    E = np.array([r["E_projected_m"] for r in rows]); rho = np.array([r["rho"] for r in rows]); cc = np.array([r["color_correct_of_24"] for r in rows])
    q = lambda a: {k: float(v) for k, v in zip(("min", "p25", "median", "p75", "max"), np.percentile(a, [0, 25, 50, 75, 100]))}
    summ = {"n_seeds": len(rows), "seeds": "20261009 (pilot) + 20261010..20261109", "E_full_m": E_full, "bar_m": bar, "rho_tolerance": RD.RHO_MAX,
            "seeds_passing_primary_rule": int(sum(r["passes_primary"] for r in rows)), "seeds_with_E_below_bar": int(sum(r["below_bar"] for r in rows)),
            "fraction_passing_primary": sum(r["passes_primary"] for r in rows) / len(rows),
            "E_projected_m": q(E), "rho": q(rho), "color_correct_of_24": {"min": int(cc.min()), "median": float(np.median(cc)), "max": int(cc.max()), "seeds_24_of_24": int(np.sum(cc == 24))},
            "groups_beating_both_baselines": {"median": float(np.median([r["groups_beating_both_baselines"] for r in rows])), "max": int(max(r["groups_beating_both_baselines"] for r in rows))},
            "J_projected_m": q([r["J_projected_m"] for r in rows]),
            "original_seed_20261009": {"E_projected_m": rows[0]["E_projected_m"], "rho": rows[0]["rho"], "rank_of_E_among_101_ascending": int(np.sum(E < E[0])) + 1},
            "caveat": "Descriptive; same 24 clips and 6 groups; no significance claim; no seed is selected as 'best'."}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(json.dumps(summ, indent=1))
    plot(rows, summ)


def plot(rows, S):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e1"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    E = np.array([r["E_projected_m"] for r in rows]); cc = np.array([r["color_correct_of_24"] for r in rows])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.9), facecolor=SURF); fig.subplots_adjust(left=0.07, right=0.98, bottom=0.27, top=0.86, wspace=0.22)
    for ax in axs:
        ax.set_facecolor(SURF); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=9)
    ax = axs[0]
    ax.hist(E, bins=np.arange(np.floor(E.min()), np.ceil(E.max()) + 1, 1.0), color=ORANGE, edgecolor=SURF, lw=1.5, zorder=3)
    from matplotlib.lines import Line2D
    for x, c, ls, lw in ((S["E_full_m"], BLUE, "-", 1.4), (S["E_full_m"] * 1.10, BLUE, (0, (4, 3)), 1.4), (S["bar_m"], INK2, (0, (4, 3)), 1.4), (rows[0]["E_projected_m"], INK, "-", 1.9)):
        ax.axvline(x, color=c, lw=lw, ls=ls, zorder=4)
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=1.4, label=f"full 768-D reader ({S['E_full_m']:.1f} m)"), Line2D([], [], color=BLUE, lw=1.4, ls=(0, (4, 3)), label="full × 1.10 (ρ tolerance)"),
                       Line2D([], [], color=INK2, lw=1.4, ls=(0, (4, 3)), label=f"baseline bar ({S['bar_m']:.1f} m)"), Line2D([], [], color=INK, lw=1.9, label=f"pilot seed 20261009 ({rows[0]['E_projected_m']:.1f} m)")],
              frameon=False, fontsize=8, loc="upper right", labelcolor=INK2)
    ax.set_xlabel("Projected-reader position RMSE (m), one value per seed"); ax.set_ylabel("Number of seeds")
    ax.set_title(f"Position error across {S['n_seeds']} random 16-D projections", loc="left", fontsize=11, color=INK)
    ax = axs[1]
    vals, cnt = np.unique(cc, return_counts=True)
    ax.bar(vals, cnt, 0.7, color=ORANGE, zorder=3)
    for v, c in zip(vals, cnt): ax.text(v, c + 1, str(c), ha="center", fontsize=9, color=INK)
    ax.set_xlabel("Colour clips correct (of 24), projected reader"); ax.set_ylabel("Number of seeds"); ax.set_xticks(vals)
    ax.set_title("Colour readability across the same seeds", loc="left", fontsize=11, color=INK)
    fig.text(0.07, 0.07, f"{S['seeds_passing_primary_rule']} of {S['n_seeds']} seeds pass the primary rule (E_proj < 16.5 m and ρ ≤ 1.10). Descriptive follow-up on the same 24 clips / 6 groups; no uncertainty or significance claimed.", fontsize=8, color=INK2)
    fig.text(0.07, 0.025, "Every seed is shown; none is selected. Data: followup_f1_seed_sweep/seed_sweep.csv. The pilot's primary result is unchanged.", fontsize=8, color=INK2)
    fig.savefig(OUT / "plot_f1_seed_sweep.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    main()
