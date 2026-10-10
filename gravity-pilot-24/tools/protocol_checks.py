#!/usr/bin/env python3
"""Software-only checks of the 24-video gravity pilot protocol (no model, no video, no inference).

Recomputes every hand-checkable number stated in MATHEMATICAL_PROTOCOL v1.0 from the
protocol's own formulas, and (with --write-scene-plan) regenerates scene_plan.csv.
Exit status is non-zero if any check fails.
"""
import argparse, csv, hashlib, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
G = (9.8, 4.9)
COLORS = (("orange", (235, 100, 60)), ("blue", (60, 140, 235)))
FRAMES, FPS, SCALE = 151, 30, 1.92
OBS_IDX = [4 * l for l in range(16)]
SEED = 20261009
results = []


def check(name, ok, detail=""):
    results.append({"check": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


def scenes():
    for s in range(6):
        i, j = s % 3, s // 3
        yield s, -40 + 40 * i, 140 + 20 * j


def pos(x0, y0, g, t):
    return np.array([x0 + 2 * t, y0 - 0.5 * g * t * t])


def scene_rows():
    rows = []
    for s, x0, y0 in scenes():
        for a, g in enumerate(G):
            for c, (cn, rgb) in enumerate(COLORS):
                tgt = pos(x0, y0, g, 4.0)
                rows.append(dict(
                    clip_id=f"s{s:02d}_g{'98' if a == 0 else '49'}_{cn}", filename=f"s{s:02d}_g{'98' if a == 0 else '49'}_{cn}.mkv",
                    scene=s, x0=x0, y0=y0, gravity_idx=a, g=g, color_idx=c, color=cn,
                    rgb=" ".join(map(str, rgb)), target_x4=round(float(tgt[0]), 6), target_y4=round(float(tgt[1]), 6)))
    return rows


def ridge_dual(A, Y, lam):
    n = A.shape[0]
    Yb = Y.mean(0)
    return A.T @ np.linalg.solve(A @ A.T + n * lam * np.eye(n), Y - Yb), Yb


def ridge_primal(A, Y, lam):
    n, d = A.shape
    Yb = Y.mean(0)
    return np.linalg.solve(A.T @ A + n * lam * np.eye(d), A.T @ (Y - Yb)), Yb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-scene-plan", action="store_true")
    args = ap.parse_args()
    rows = scene_rows()

    check("24 clips, 24 unique filenames", len(rows) == 24 and len({r["filename"] for r in rows}) == 24)
    check("each scene has 4 clips; each fold 20/4 split", all(sum(r["scene"] == f for r in rows) == 4 for f in range(6)))
    check("training set per fold is 10 orange / 10 blue / 10 per gravity",
          all(sum(r["color"] == "orange" for r in rows if r["scene"] != f) == 10 and
              sum(r["gravity_idx"] == 0 for r in rows if r["scene"] != f) == 10 for f in range(6)))
    check("six start positions as stated",
          [(x, y) for _, x, y in scenes()] == [(-40, 140), (0, 140), (40, 140), (-40, 160), (0, 160), (40, 160)])
    check("151 frames, last timestamp 5 s, playback 151/30", FRAMES == 151 and (FRAMES - 1) / FPS == 5.0 and abs(FRAMES / FPS - 5.0333333) < 1e-6)
    check("observed indices 0..60 step 4, t_l=2l/15, last t=2 s",
          OBS_IDX[-1] == 60 and all(abs(k / FPS - 2 * l / 15) < 1e-12 for l, k in enumerate(OBS_IDX)) and len(OBS_IDX) == 16)
    check("target frame 120 is t=4 s", 120 / FPS == 4.0)

    # worked example s=0
    t0 = pos(-40, 140, 9.8, 4.0), pos(-40, 140, 4.9, 4.0)
    check("worked check s=0 targets (-32,61.6),(-32,100.8)",
          np.allclose(t0[0], (-32, 61.6)) and np.allclose(t0[1], (-32, 100.8)))
    check("s=0 baseline-gravity y(5)=17.5", math.isclose(pos(-40, 140, 9.8, 5.0)[1], 17.5))

    # visibility / pixel mapping
    ext = np.array([pos(x0, y0, g, t) for _, x0, y0 in scenes() for g in G for t in np.linspace(0, 5, 151)])
    check("centres lie in x[-40,50], y[17.5,160]",
          math.isclose(ext[:, 0].min(), -40) and math.isclose(ext[:, 0].max(), 50) and math.isclose(ext[:, 1].min(), 17.5) and math.isclose(ext[:, 1].max(), 160))
    u, v = SCALE * (ext[:, 0] + 100), SCALE * (200 - ext[:, 1])
    check("ball (r=9.6 px) fully inside 384x384 for all frames",
          (u - 9.6).min() >= 0 and (u + 9.6).max() <= 384 and (v - 9.6).min() >= 0 and (v + 9.6).max() <= 384,
          f"u[{u.min():.1f},{u.max():.1f}] v[{v.min():.1f},{v.max():.1f}]")
    check("5 m radius -> 9.6 px; 200 m -> 384 px", math.isclose(5 * SCALE, 9.6) and math.isclose(200 * SCALE, 384))
    check("encoder tokens 8x24x24 = 4608", (16 // 2) * (384 // 16) ** 2 == 4608)

    # paired physical targets
    by = {}
    for r in rows:
        by.setdefault((r["scene"], r["gravity_idx"]), set()).add((r["target_x4"], r["target_y4"]))
    check("paired colour clips share identical true targets", all(len(v) == 1 for v in by.values()))

    # baselines
    ta, tb = 56 / 30, 2.0
    ecv = {}
    for g in G:
        vh = (pos(0, 0, g, tb) - pos(0, 0, g, ta)) / (tb - ta)
        pcv = pos(0, 0, g, tb) + 2 * vh
        ecv[g] = np.linalg.norm(pcv - pos(0, 0, g, 4.0))
    check("CV error 32/15*g: 20.9066667 / 10.4533333",
          math.isclose(ecv[9.8], 32 / 15 * 9.8) and abs(ecv[9.8] - 20.9066667) < 1e-7 and abs(ecv[4.9] - 10.4533333) < 1e-7)
    E_cv = math.sqrt((ecv[9.8] ** 2 + ecv[4.9] ** 2) / 2)
    d = np.mean([pos(0, 0, g, 4.0) - pos(0, 0, g, 0.0) for g in G for _ in range(10)], axis=0)
    check("prior displacement (8,-58.8)", np.allclose(d, (8, -58.8)))
    ep = [np.linalg.norm(pos(0, 0, g, 0) + d - pos(0, 0, g, 4)) for g in G]
    E_prior = math.sqrt(np.mean(np.square(ep)))
    check("prior error 19.6 m for both gravities", np.allclose(ep, 19.6))
    check("aggregate RMSE = sqrt(mean of squared errors), not mean of errors", math.isclose(E_cv, 10.4533333 * math.sqrt(2.5), rel_tol=1e-6), f"E_CV={E_cv:.6f}")
    print(f"INFO E_CV={E_cv:.6f} m, E_prior={E_prior:.6f} m, min={min(E_cv, E_prior):.6f} m (CV binds in every group)")
    check("FINDING: min(E_CV,E_prior)=E_CV, so 'beat both' reduces to 'beat CV'", E_cv < E_prior)

    # storage
    check("storage: 3072 B, 64 B, 48x, 97.9167%, 73728, 1536, 49152, 50688",
          768 * 4 == 3072 and 16 * 4 == 64 and 3072 // 64 == 48 and round((1 - 16 / 768) * 100, 4) == 97.9167
          and 24 * 3072 == 73728 and 24 * 64 == 1536 and 16 * 768 * 4 == 49152 and 49152 + 1536 == 50688)

    # ridge: dual == primal on tiny example, unpenalised intercept, scalar RMS scale
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(20, 7)) * 3 + 5
    mu = raw.mean(0)
    s_f = math.sqrt(((raw - mu) ** 2).sum(1).mean())
    A = (raw - mu) / s_f
    check("scalar RMS scaling gives unit total training variance", math.isclose((A ** 2).sum(1).mean(), 1.0))
    Y = rng.normal(size=(20, 2)) + 10
    Wd, Yb = ridge_dual(A, Y, 0.01)
    Wp, _ = ridge_primal(A, Y, 0.01)
    check("dual ridge == primal ridge", np.allclose(Wd, Wp, atol=1e-10))
    check("unpenalised intercept: mean fitted = mean target", np.allclose((A @ Wd + Yb).mean(0), Y.mean(0)))
    Ys = Y + 1000.0
    Wd2, Yb2 = ridge_dual(A, Ys, 0.01)
    check("target offset moves only the intercept", np.allclose(Wd, Wd2) and np.allclose(Yb2 - Yb, 1000.0))
    check("tie-to-orange: score 0 -> orange (c=1[score>0])", int(0.0 > 0) == 0)

    # projection
    R = ((2 * np.random.Generator(np.random.PCG64(SEED)).integers(0, 2, size=(16, 768)) - 1) / 4).astype(np.float32)
    h = hashlib.sha256(R.tobytes()).hexdigest()
    R2 = ((2 * np.random.Generator(np.random.PCG64(SEED)).integers(0, 2, size=(16, 768)) - 1) / 4).astype(np.float32)
    check("projection reproducible within one environment", h == hashlib.sha256(R2.tobytes()).hexdigest(), f"numpy {np.__version__} sha256 {h}")
    check("projection entries are +/-0.25; 1/sqrt(16)=1/4", set(np.unique(R).tolist()) == {-0.25, 0.25})
    zs = rng.normal(size=(20, 768))
    Rz = zs @ R.T
    check("projection scale 1/sqrt(16) is removed by scalar RMS scaling (note)",
          np.allclose(((Rz - Rz.mean(0)) / math.sqrt(((Rz - Rz.mean(0)) ** 2).sum(1).mean())),
                      ((4 * Rz - (4 * Rz).mean(0)) / math.sqrt(((4 * Rz - (4 * Rz).mean(0)) ** 2).sum(1).mean()))))
    print(f"INFO projection_sha256 ({np.__version__}) = {h}")

    if args.write_scene_plan:
        out = HERE / "scene_plan.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print("wrote", out)
    (HERE / "tools" / "protocol_checks_result.json").write_text(json.dumps(
        {"numpy": np.__version__, "projection_sha256_reference": h, "E_CV": E_cv, "E_prior": E_prior, "checks": results}, indent=1))
    sys.exit(0 if all(r["pass"] for r in results) else 1)


if __name__ == "__main__":
    main()
