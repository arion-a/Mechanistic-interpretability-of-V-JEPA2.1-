#!/usr/bin/env python3
"""Stage B of the 24-video gravity pilot: selftest / extract / analyze / report.
Every command except `selftest` refuses to run without an approval receipt matching the manifest SHA-256.
Fixed choices only (protocol v1.0 + dataset v1.1): one checkpoint, final layer, seed 20261009, 16-D projection, lambda=0.01.
"""
import argparse, csv, hashlib, json, math, platform, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import pilot as P  # noqa: E402  (pure functions only; no side effects on import)
import readers as RD  # noqa: E402

DS, OUT = P.DS, ROOT / "results"
REPO = ROOT / "third_party" / "vjepa2"
CKPT = ROOT / "cache" / "vjepa2_1_vitb_dist_vitG_384.pt"
REVISION = "204698b45b3712590f06245fbfba32d3be539812"
SEED = 20261009
CODE_FILES = ["pilot.py", "readers.py", "stage_b.py", "report_plots.py"]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def code_hash():
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update((ROOT / f).read_bytes())
    return h.hexdigest()


def require_receipt():
    ok, why = P.receipt_ok()
    if not ok:
        sys.exit(f"REFUSED: dataset not approved ({why}). Stage B is locked.")
    return json.loads((DS / "approval_receipt.json").read_text())


def manifest():
    return json.loads((DS / "manifest.json").read_text())


# ------------------------------------------------------------------ projection
def make_projection():
    b = np.random.Generator(np.random.PCG64(SEED)).integers(0, 2, size=(16, 768))
    return ((2 * b - 1) / 4).astype(np.float32)


# ------------------------------------------------------------------ selftest (software-only; no model)
def cmd_selftest(_=None):
    res = []

    def chk(n, ok, d=""):
        res.append((n, bool(ok), d)); print(("PASS " if ok else "FAIL ") + n + (f" [{d}]" if d else ""))

    rng = np.random.default_rng(7)
    X, Y = rng.normal(size=(24, 30)) * 3 + 2, rng.normal(size=(24, 2))
    tr = np.arange(24) >= 4
    # no held-out row contributes to means/scales/weights
    _, m1 = RD.fit_predict(X[tr], Y[tr], X[~tr])
    X2, Y2 = X.copy(), Y.copy(); X2[~tr] += 1e3; Y2[~tr] -= 1e3
    _, m2 = RD.fit_predict(X2[tr], Y2[tr], X2[~tr])
    chk("held-out rows do not affect training mean/scale/weights/intercept",
        all(np.array_equal(m1[k], m2[k]) for k in ("mean", "scale", "W", "Ybar")))
    # dual == primal, unpenalised intercept, linear solve
    mean, s = RD.fit_scaler(X[tr]); A = (X[tr] - mean) / s
    Wd, Yb = RD.ridge_dual(A, Y[tr]); n = A.shape[0]
    Wp = np.linalg.solve(A.T @ A + n * RD.LAMBDA * np.eye(A.shape[1]), A.T @ (Y[tr] - Yb))
    chk("dual ridge == primal ridge on a tiny example", np.allclose(Wd, Wp, atol=1e-9))
    chk("intercept unpenalised (constant target recovered exactly)", np.allclose(RD.fit_predict(X[tr], np.full((20, 1), 5.0), X[~tr])[0], 5.0))
    chk("scalar RMS scale -> unit total training variance", math.isclose(((A) ** 2).sum(1).mean(), 1.0))
    # degenerate scale
    try:
        RD.fit_scaler(np.ones((20, 5))); chk("zero-variance fold stops as degenerate", False)
    except RD.Degenerate:
        chk("zero-variance fold stops as degenerate (no silent replacement)", True)
    chk("tie-to-orange: score 0 -> orange(0), >0 -> blue(1)", list(RD.color_label([0.0, 1e-12, -1e-12])) == [0, 1, 0])
    # aggregate RMSE vs pooled squared errors (not mean of group RMSEs)
    e = np.array([1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0])
    chk("overall RMSE = sqrt(pooled mean squared error), differs from mean of group RMSEs",
        math.isclose(RD.euclid_rmse(e), math.sqrt((1 + 25) / 2)) and not math.isclose(RD.euclid_rmse(e), 3.0))
    chk("rho zero-denominator handled (E_full<=1e-9 -> undefined, with reason)", RD.rho(1.0, 1e-10)[0] is None and RD.rho(1.1, 1.0)[0] == 1.1)
    # projection hash reproducible
    R1, R2 = make_projection(), make_projection()
    chk("projection reproducible; (16,768) float32 entries +/-0.25", R1.shape == (16, 768) and R1.dtype == np.float32 and np.array_equal(R1, R2)
        and set(np.unique(R1).tolist()) == {-0.25, 0.25}, sha256_bytes(R1.tobytes()))
    # features never include labels/metadata: embedding arrays have fixed width and come only from the encoder input tensor
    chk("encoder input depends only on frames 0,4,...,60 (modify 61-270)", True if _input_property() else False)
    ok = all(r[1] for r in res)
    print("SELFTEST", "PASSED" if ok else "FAILED")
    if not ok:
        sys.exit(1)
    return ok


def _input_property():
    fr = P.ffmpeg_decode(DS / manifest()["clips"][0]["filename"])
    mod = fr.copy(); mod[61:] = np.random.default_rng(0).integers(0, 256, size=mod[61:].shape, dtype=np.uint8)
    return np.array_equal(P.build_encoder_input(fr), P.build_encoder_input(mod))


# ------------------------------------------------------------------ extract
def load_encoder():
    import torch
    sys.path.insert(0, str(REPO))
    from src.hub.backbones import vjepa2_1_vit_base_384, _clean_backbone_key
    torch.set_grad_enabled(False)
    enc, _pred = vjepa2_1_vit_base_384(pretrained=False)  # architecture only; weights loaded explicitly below
    try:
        ck = torch.load(CKPT, map_location="cpu", weights_only=True)
        wo = True
    except Exception as e:  # pragma: no cover - recorded in provenance if it ever happens
        sys.exit(f"BLOCKER: weights_only load failed ({e!r}); not falling back to unsafe unpickling without user decision")
    keys = sorted(ck.keys())
    sd = _clean_backbone_key(ck["ema_encoder"])
    r = enc.load_state_dict(sd, strict=True)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, dict(checkpoint_top_level_keys=keys, weights_only_load=wo, load_result=str(r), n_encoder_params=sum(p.numel() for p in enc.parameters()))


def cmd_extract(_):
    receipt = require_receipt()
    cmd_selftest()
    import torch
    man = manifest()
    ck_hash = P.sha256_file(CKPT)
    repo_rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert repo_rev == REVISION, f"source revision {repo_rev} != pinned {REVISION}"
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    assert not dirty, "vjepa2 checkout has local modifications"
    for c in man["clips"]:
        assert P.sha256_file(DS / c["filename"]) == c["sha256"], f"video hash mismatch {c['clip_id']}"
    enc, load_info = load_encoder()
    torch.set_num_threads(4)
    cache = ROOT / "cache" / "emb"; cache.mkdir(parents=True, exist_ok=True)
    base_key = dict(code=code_hash(), config=P.sha256_file(ROOT / "pilot_config.json"), revision=REVISION, ckpt=ck_hash)
    Z, rows, t_start = [], [], time.time()
    for n, c in enumerate(man["clips"]):
        frames = P.ffmpeg_decode(DS / c["filename"])
        x = P.build_encoder_input(frames)
        assert x.shape == (1, 3, 16, 384, 384) and x.dtype == np.float32
        in_hash = sha256_bytes(x.tobytes())
        key = dict(base_key, video=c["sha256"], input_idx=P.OBS_IDX, input_tensor=in_hash)
        kf = cache / f"{c['clip_id']}.json"
        if kf.exists() and json.loads(kf.read_text()) == key and (cache / f"{c['clip_id']}.npy").exists():
            z = np.load(cache / f"{c['clip_id']}.npy"); src = "cache(matching provenance)"
        else:
            t0 = time.time()
            tok = enc(torch.from_numpy(x))  # float32, full 16-frame input only
            assert tuple(tok.shape) == (1, 4608, 768), f"unexpected token shape {tuple(tok.shape)} (blocker, not permission to pool another output)"
            assert torch.isfinite(tok).all()
            z = tok[0].mean(0).numpy().astype(np.float32)
            np.save(cache / f"{c['clip_id']}.npy", z); kf.write_text(json.dumps(key, sort_keys=True)); src = f"computed {time.time() - t0:.1f}s"
        assert z.shape == (768,) and np.isfinite(z).all()
        Z.append(z)
        rows.append(dict(row=n, clip_id=c["clip_id"], scene=c["scene"], g=c["g"], color=c["color"], video_sha256=c["sha256"],
                         input_frame_indices=" ".join(map(str, P.OBS_IDX)), input_tensor_sha256=in_hash))
        print(f"[{n + 1:2d}/24] {c['clip_id']}  {src}", flush=True)
    Z = np.stack(Z).astype(np.float32)
    np.save(OUT / "embeddings_full.npy", Z)
    R = make_projection()
    np.save(OUT / "projection.npy", R)
    U = (Z @ R.T).astype(np.float32)
    np.save(OUT / "embeddings_projected.npy", U)
    with open(OUT / "embedding_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    prov = dict(stage="B-extract", protocol_version="1.0", dataset_version=man["dataset_version"], manifest_sha256=P.sha256_file(DS / "manifest.json"),
                approval_receipt=receipt, source_repo="https://github.com/facebookresearch/vjepa2", source_revision=repo_rev,
                checkpoint_file=CKPT.name, checkpoint_sha256=ck_hash, checkpoint_size_bytes=CKPT.stat().st_size, checkpoint_key="ema_encoder", **load_info,
                model_entry="src.hub.backbones.vjepa2_1_vit_base_384(pretrained=False) architecture + explicit strict load of ema_encoder; predictor not loaded; hub default num_frames=64 RoPE with interpolate_rope=True",
                output="encoder forward default path: final post-LayerNorm tokens (norms_block[-1]); hierarchical output disabled; shape [1,4608,768]",
                pooling="mean over 4608 tokens, float32, no L2 norm", input_frame_indices=P.OBS_IDX, input_tensor_shape=[1, 3, 16, 384, 384],
                precision="float32", device="cpu", torch_threads=4, torch=torch.__version__, numpy=np.__version__, python=sys.version, platform=platform.platform(),
                code_sha256=code_hash(), config_sha256=base_key["config"], projection_seed=SEED, projection_sha256=sha256_bytes(R.tobytes()),
                embeddings_full_sha256=sha256_bytes(Z.tobytes()), embeddings_projected_sha256=sha256_bytes(U.tobytes()),
                git_sha=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
                total_seconds=round(time.time() - t_start, 1))
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print("extract complete:", Z.shape, U.shape, "projection sha256", prov["projection_sha256"])


# ------------------------------------------------------------------ analyze
def cmd_analyze(_):
    require_receipt()
    man = manifest(); clips = man["clips"]
    prov = json.loads((OUT / "provenance.json").read_text())
    # resume/provenance guard: embeddings must come from this exact dataset/code/config/checkpoint
    assert prov["manifest_sha256"] == P.sha256_file(DS / "manifest.json"), "embeddings do not match current manifest"
    assert prov["config_sha256"] == P.sha256_file(ROOT / "pilot_config.json") and prov["source_revision"] == REVISION
    Z = np.load(OUT / "embeddings_full.npy"); U = np.load(OUT / "embeddings_projected.npy"); R = np.load(OUT / "projection.npy")
    assert Z.shape == (24, 768) and U.shape == (24, 16) and R.shape == (16, 768) and np.isfinite(Z).all() and np.isfinite(U).all()
    assert sha256_bytes(Z.tobytes()) == prov["embeddings_full_sha256"] and sha256_bytes(R.tobytes()) == prov["projection_sha256"]
    assert np.array_equal(U, (Z @ R.T).astype(np.float32))
    n = 24
    scene = np.array([c["scene"] for c in clips]); gi = np.array([c["gravity_idx"] for c in clips]); col = np.array([c["color_idx"] for c in clips])
    gs = np.array([c["g"] for c in clips]); P0 = np.array([[c["x0"], c["y0"]] for c in clips], float)
    T = np.array([c["target_t4_m"] for c in clips], float)
    reps = {"full": Z.astype(np.float64), "projected": U.astype(np.float64)}
    fitdir = OUT / "fitted_readers"; fitdir.mkdir(exist_ok=True)
    pos = {k: np.full((n, 2), np.nan) for k in reps}; score = {k: np.full(n, np.nan) for k in reps}
    degenerate = []; fold_of = scene.copy()
    for f in range(6):
        tr, te = scene != f, scene == f
        for rname, X in reps.items():
            try:
                pp, mp = RD.fit_predict(X[tr], T[tr] / 100.0, X[te])
                cc, mc = RD.fit_predict(X[tr], np.where(col[tr] == 1, 1.0, -1.0)[:, None], X[te])
            except RD.Degenerate as e:
                degenerate.append(dict(fold=f, representation=rname, reason=str(e))); continue
            pos[rname][te] = 100.0 * pp; score[rname][te] = cc[:, 0]
            for task, m in (("position", mp), ("color", mc)):
                np.savez(fitdir / f"fold{f}_{rname}_{task}.npz", train_idx=np.where(tr)[0], test_idx=np.where(te)[0], train_clips=np.array([clips[i]["clip_id"] for i in np.where(tr)[0]]),
                         test_clips=np.array([clips[i]["clip_id"] for i in np.where(te)[0]]), feature_mean=m["mean"], feature_scale=m["scale"], weights=m["W"], target_mean=m["Ybar"], ridge_lambda=RD.LAMBDA)
    assert not degenerate, f"degenerate fold/representation: {degenerate}"
    assert all(np.isfinite(v).all() for v in pos.values()) and all(np.isfinite(v).all() for v in score.values())
    # baselines (exact simulator centres; labelled oracle-observation)
    ta, tb = 56 / 30, 2.0
    cv = np.zeros((n, 2)); prior = np.zeros((n, 2))
    for i, c in enumerate(clips):
        pa, pb = np.array(P.center(c["x0"], c["y0"], c["g"], ta)), np.array(P.center(c["x0"], c["y0"], c["g"], tb))
        cv[i] = pb + 2.0 * (pb - pa) / (tb - ta)
    for f in range(6):
        tr = scene != f
        dbar = (T[tr] - P0[tr]).mean(0)
        prior[scene == f] = P0[scene == f] + dbar
    preds = {"full": pos["full"], "projected": pos["projected"], "cv": cv, "prior": prior}
    err = {k: np.linalg.norm(v - T, axis=1) for k, v in preds.items()}
    rm = RD.euclid_rmse
    def grp(k, mask): return rm(err[k][mask])
    summ = {"dataset_manifest_sha256": prov["manifest_sha256"], "protocol": "v1.0 maths on dataset v1.1", "n_clips": n, "units": "metres for all position errors"}
    E = {k: rm(err[k]) for k in preds}
    summ["pooled_position_rmse_m"] = E
    summ["pooled_xy_rmse_m"] = {k: {"x": float(np.sqrt(np.mean((preds[k][:, 0] - T[:, 0]) ** 2))), "y": float(np.sqrt(np.mean((preds[k][:, 1] - T[:, 1]) ** 2)))} for k in preds}
    summ["group_position_rmse_m"] = {k: {str(f): grp(k, scene == f) for f in range(6)} for k in preds}
    summ["gravity_position_rmse_m"] = {k: {str(g): grp(k, gs == g) for g in (9.8, 4.9)} for k in preds}
    base = min(E["cv"], E["prior"])
    summ["baseline_bar_min_E_cv_E_prior_m"] = base
    summ["gain_vs_cv_G"] = {k: 1 - E[k] / E["cv"] for k in ("full", "projected")}
    r, why = RD.rho(E["projected"], E["full"])
    summ["rho_projected_over_full"] = {"value": r, "null_reason": why, "absolute_change_m": E["projected"] - E["full"]}
    summ["rho_by_fold"] = {str(f): (RD.rho(grp("projected", scene == f), grp("full", scene == f))[0]) for f in range(6)}
    beat_group = {k: [bool(grp(k, scene == f) < min(grp("cv", scene == f), grp("prior", scene == f))) for f in range(6)] for k in ("full", "projected")}
    summ["breadth_b_r"] = {k: {"fraction_groups_beating_both_baselines": sum(v) / 6, "per_group": {str(f): v[f] for f in range(6)}} for k, v in beat_group.items()}
    # colour
    pc = {k: RD.color_label(score[k]) for k in reps}
    A = {k: float(np.mean(pc[k] == col)) for k in reps}
    cm = lambda k: [[int(np.sum((col == a) & (pc[k] == b))) for b in (0, 1)] for a in (0, 1)]
    summ["color"] = {"correct_of_24": {k: int(np.sum(pc[k] == col)) for k in reps}, "accuracy": A,
                     "recall": {k: {"orange": float(np.mean(pc[k][col == 0] == 0)), "blue": float(np.mean(pc[k][col == 1] == 1))} for k in reps},
                     "confusion_matrix_rows_true_orange_blue_cols_pred_orange_blue": {k: cm(k) for k in reps},
                     "group_accuracy": {k: {str(f): float(np.mean(pc[k][scene == f] == col[scene == f])) for f in range(6)} for k in reps},
                     "delta_A_full_minus_projected": A["full"] - A["projected"], "chance_reference": 0.5}
    # paired colour sensitivity J (12 pairs: scene x gravity)
    J = {}
    for k in reps:
        d = []
        for s in range(6):
            for a in range(2):
                i0 = int(np.where((scene == s) & (gi == a) & (col == 0))[0][0]); i1 = int(np.where((scene == s) & (gi == a) & (col == 1))[0][0])
                d.append(np.sum((pos[k][i1] - pos[k][i0]) ** 2))
        J[k] = float(np.sqrt(np.mean(d)))
    summ["paired_color_sensitivity_J_m"] = J
    # storage (reported separately from error)
    summ["storage_bytes"] = {"full_per_clip": Z[0].nbytes, "projected_per_clip": U[0].nbytes, "full_24": Z.nbytes, "projected_24": U.nbytes,
                             "reduction_factor": Z[0].nbytes / U[0].nbytes, "percent_fewer_values": 100 * (1 - 16 / 768),
                             "projection_matrix_overhead": R.nbytes, "projected_dataset_incl_matrix": U.nbytes + R.nbytes,
                             "note": "payload at float32; not Shannon information; encoder/reader/provenance storage excluded"}
    # decision flags (rules fixed in protocol section 9)
    prim = bool(E["full"] < base and E["projected"] < base and r is not None and r <= RD.RHO_MAX)
    summ["decision"] = {"primary_prediction_retention": {"E_full_lt_bar": bool(E["full"] < base), "E_projected_lt_bar": bool(E["projected"] < base),
                                                           "rho_le_1.10": (None if r is None else bool(r <= RD.RHO_MAX)), "PASS": prim},
                        "selective_accessibility": {"A_full_gt_0.5": bool(A["full"] > 0.5), "deltaA_ge_0.125": bool(A["full"] - A["projected"] >= 0.125 - 1e-12),
                                                    "PATTERN_HOLDS": bool(A["full"] > 0.5 and A["full"] - A["projected"] >= 0.125 - 1e-12)},
                        "no_significance_claims": True}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    # predictions.csv
    with open(OUT / "predictions.csv", "w", newline="") as f:
        cols = ["row", "fold", "clip_id", "scene", "gravity", "color", "true_x", "true_y", "full_x", "full_y", "projected_x", "projected_y", "cv_x", "cv_y", "prior_x", "prior_y",
                "err_full_m", "err_projected_m", "err_cv_m", "err_prior_m", "color_true", "color_score_full", "color_score_projected", "color_pred_full", "color_pred_projected"]
        w = csv.writer(f); w.writerow(cols)
        for i, c in enumerate(clips):
            w.writerow([i, int(fold_of[i]), c["clip_id"], c["scene"], c["g"], c["color"], *T[i], *pos["full"][i], *pos["projected"][i], *cv[i], *prior[i],
                        err["full"][i], err["projected"][i], err["cv"][i], err["prior"][i], c["color_idx"], score["full"][i], score["projected"][i], int(pc["full"][i]), int(pc["projected"][i])])
    print(json.dumps({"E": E, "rho": summ["rho_projected_over_full"], "color_correct": summ["color"]["correct_of_24"], "J": J, "decision": summ["decision"]}, indent=1))


# ------------------------------------------------------------------ report
def cmd_report(_):
    require_receipt()
    S = json.loads((OUT / "summary.json").read_text())
    rows = list(csv.DictReader(open(OUT / "predictions.csv")))
    assert len(rows) == 24 and len({r["clip_id"] for r in rows}) == 24
    # every number must be reproducible from predictions.csv alone
    chk = []
    for k in ("full", "projected", "cv", "prior"):
        e = np.array([float(r[f"err_{k}_m"]) for r in rows])
        d = np.array([[float(r[f"{k}_x"]) - float(r["true_x"]), float(r[f"{k}_y"]) - float(r["true_y"])] for r in rows])
        chk.append(abs(RD.euclid_rmse(e) - S["pooled_position_rmse_m"][k]) < 1e-9 and np.allclose(np.hypot(d[:, 0], d[:, 1]), e))
        for f in range(6):
            chk.append(abs(RD.euclid_rmse([float(r[f"err_{k}_m"]) for r in rows if int(r["scene"]) == f]) - S["group_position_rmse_m"][k][str(f)]) < 1e-9)
    for k in ("full", "projected"):
        chk.append(sum(int(r[f"color_pred_{k}"]) == int(r["color_true"]) for r in rows) == S["color"]["correct_of_24"][k])
    assert all(chk), "summary.json is not reproducible from predictions.csv"
    print(f"reproducibility: {len(chk)} summary values recomputed from predictions.csv - all match")
    import report_plots
    report_plots.make_all(OUT)
    print("plots written to", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n, f in (("selftest", cmd_selftest), ("extract", cmd_extract), ("analyze", cmd_analyze), ("report", cmd_report)):
        sp.add_parser(n).set_defaults(fn=f)
    a = ap.parse_args(); a.fn(a)
