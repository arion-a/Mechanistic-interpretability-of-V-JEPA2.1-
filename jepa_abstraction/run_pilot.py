"""End-to-end pilot: pretrain JEPA -> frozen features -> equal-budget readouts -> ARS -> probes -> causal swap.

Usage: python run_pilot.py --out results/pilot [--c 8 --h 16 --steps 1500 ...]
PILOT output only: not a confirmatory result (see docs/jepa-abstraction-experiment).
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn
from jepa_abs import sim, metrics, model as M


def pretrain(bun, a, cfg, log):
    net = M.JEPA(cfg.res, a.patch, a.d, a.depth, max_t=a.c, max_off=a.h)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    N = bun.n_groups * bun.K
    ctx_pos, _ = bun.clips(a.c, [1])
    p0_all, v0_all, g_all = bun.expand(bun.p0), bun.expand(bun.v0), bun.g()
    rng = np.random.default_rng(a.seed + 1)
    for step in range(a.steps):
        idx = rng.integers(0, N, a.batch)
        off = rng.integers(1, a.h + 1, a.batch)
        ctx = torch.from_numpy(sim.render_ctx(bun, ctx_pos, cfg, idx))
        # target frame at per-sample future offset (law labels never enter the loader)
        tp = np.stack([sim.positions(p0_all[i:i + 1], v0_all[i:i + 1], g_all[i:i + 1],
                                     sim.frame_tau([sim.ANCHOR + o]))[0, 0] for i, o in zip(idx, off)])
        nuis = {k: bun.expand(v)[idx] for k, v in bun.nuis.items()}
        tgt = torch.from_numpy(sim.render(tp[:, None], nuis, cfg)[:, 0])
        loss, _, tok = net.loss(ctx, tgt, torch.from_numpy(off).long())
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        net.ema(0.996 + (0.9999 - 0.996) * step / max(1, a.steps - 1))
        if step % 100 == 0 or step == a.steps - 1:
            f = tok.mean(1)
            log.append(dict(step=step, loss=float(loss), feat_std=float(f.std(0).median()), eff_rank=M.effective_rank_fraction(f)))
            print(log[-1], flush=True)
    return net


@torch.no_grad()
def features(net, bun, a, cfg):
    ctx_pos, fut = bun.clips(a.c, np.arange(1, a.h + 1))
    zs, layers = [], []
    for i in range(0, len(ctx_pos), 256):
        idx = np.arange(i, min(i + 256, len(ctx_pos)))
        z, L = net.z(torch.from_numpy(sim.render_ctx(bun, ctx_pos, cfg, idx)))
        zs.append(z); layers.append(L)
    return torch.cat(zs).numpy(), torch.cat(layers).numpy(), fut


def const_velocity(bun, h):
    tau = sim.frame_tau(sim.ANCHOR + np.arange(1, h + 1))
    return bun.expand(bun.p0)[:, None] + bun.expand(bun.v0)[:, None] * tau[None, :, None]


def fit_readout(Xtr, Ytr, Xva, Yva, seed, max_updates=4000):
    """Equal-budget MLP: 2x256 ReLU, AdamW, batch 128, val early stopping, lr in {1e-4,1e-3}."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    ymu, ysd = Ytr.mean(0), Ytr.std(0) + 1e-6
    f = lambda X: torch.tensor((X - mu) / sd, dtype=torch.float32)
    xt, yt, xv = f(Xtr), torch.tensor((Ytr - ymu) / ysd, dtype=torch.float32), f(Xva)
    best = None
    for lr in (1e-4, 1e-3):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, Ytr.shape[1]))
        opt = torch.optim.AdamW(net.parameters(), lr=lr)
        bv, bad, bst = 1e9, 0, None
        g = torch.Generator().manual_seed(seed)
        for u in range(max_updates):
            i = torch.randint(0, len(xt), (128,), generator=g)
            l = ((net(xt[i]) - yt[i]) ** 2).mean(); opt.zero_grad(); l.backward(); opt.step()
            if u % 200 == 0:
                with torch.no_grad(): v = float(((net(xv).numpy() * ysd + ymu - Yva) ** 2).mean())
                if v < bv: bv, bad, bst = v, 0, {k: t.clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 10: break
        if best is None or bv < best[0]:
            net.load_state_dict(bst); best = (bv, net)
    net = best[1].eval()
    def predict(X):
        with torch.no_grad(): return net(f(X)).numpy() * ysd + ymu
    return predict


def group_loss(pred, fut, K, n_groups):
    """Position MSE / L_ref^2 (L_ref=1 m): sum xyz, mean over offsets; laws equal-weighted within a group."""
    se = ((pred - fut) ** 2).sum(-1).mean(-1)
    return se.reshape(n_groups, K).mean(1)


def ridge_r2(Xtr, ytr, Xte, yte, lam=1e-2):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    w = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (ytr - ytr.mean()))
    p = B @ w + ytr.mean()
    return float(1 - ((p - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    for k, v in dict(c=8, h=16, steps=1500, batch=32, lr=3e-4, d=64, depth=2, patch=8, res=32,
                     n_pre=1500, n_read=600, n_val=200, n_test=300, seed=0).items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--out", default="results/pilot")
    a = ap.parse_args()
    np.random.seed(a.seed); torch.manual_seed(a.seed); os.makedirs(a.out, exist_ok=True)
    cfg = sim.RenderCfg(res=a.res); t0 = time.time()
    # disjoint corpora from distinct generator roots: pretrain / readout-train / validation / test
    pre, rd, va, te = (sim.make_bundle(a.seed * 1000 + i, n) for i, n in enumerate([a.n_pre, a.n_read, a.n_val, a.n_test]))
    bun = dict(rd=rd, va=va, te=te)
    K, h = te.K, a.h
    log = []
    net = pretrain(pre, a, cfg, log)
    rand = M.JEPA(a.res, a.patch, a.d, a.depth, max_t=a.c, max_off=a.h)     # random-encoder control
    net.eval(); rand.eval()
    res = dict(args=vars(a), pretrain_log=log)
    feats = {n: {s: features(e, b, a, cfg) for s, b in bun.items()} for n, e in (("jepa", net), ("random", rand))}
    S = {s: b.state() for s, b in bun.items()}
    G = {s: b.g() for s, b in bun.items()}
    fut = {s: feats["jepa"][s][2] for s in bun}
    cv = {s: const_velocity(b, h) for s, b in bun.items()}
    tau = sim.frame_tau(sim.ANCHOR + np.arange(1, h + 1))

    def analytic(g):                       # exact free-flight prediction given a gravity value per clip
        out = cv["te"].copy(); out[..., 2] -= 0.5 * g[:, None] * tau[None] ** 2; return out
    # primary baselines: Bayes state predictor (prior-mean g) and law oracle (true g)
    L_S = group_loss(analytic(np.full_like(G["te"], rd.laws.mean())), fut["te"], K, te.n_groups)
    L_O = group_loss(analytic(G["te"]), fut["te"], K, te.n_groups)
    res["oracle_residual_max"] = float(L_O.max())

    # readouts predict the residual from constant-velocity extrapolation (a fixed function of s, same for all)
    Y = {s: (fut[s] - cv[s]).reshape(len(fut[s]), -1) for s in bun}
    def readout(kind, enc):
        X = {s: np.concatenate([S[s]] + ([feats[enc][s][0]] if kind == "z" else [G[s][:, None]] if kind == "g" else []), 1) for s in bun}
        return fit_readout(X["rd"], Y["rd"], X["va"], Y["va"], a.seed), X["te"]
    to_pos = lambda p: p.reshape(-1, h, 3) + cv["te"]
    R = {n: readout(*spec) for n, spec in dict(state=("s", "jepa"), jepa=("z", "jepa"), random_enc=("z", "random"),
                                               learned_oracle=("g", "jepa")).items()}
    L = {n: group_loss(to_pos(p(X)), fut["te"], K, te.n_groups) for n, (p, X) in R.items()}
    # shuffled-history control: permute z across anchors within each law (keeps law marginal, breaks pairing)
    pr, Xte = R["jepa"]; rng = np.random.default_rng(a.seed + 7)
    zte = feats["jepa"]["te"][0]; Xsh = Xte.copy()
    for k in range(K):
        ix = np.arange(k, len(zte), K); Xsh[ix, 6:] = zte[rng.permutation(ix)]
    L["shuffled_history"] = group_loss(to_pos(pr(Xsh)), fut["te"], K, te.n_groups)
    res["mean_losses"] = {k: float(v.mean()) for k, v in L.items()} | dict(analytic_state=float(L_S.mean()), analytic_oracle=float(L_O.mean()))
    for k in ("jepa", "random_enc", "shuffled_history", "state", "learned_oracle"):
        res[f"ARS_{k}"] = metrics.evaluate(L_S, L[k], L_O, seed=a.seed)

    # inferability probes of gravity per layer (readout-split train, held-out-group test), with/without state
    res["probe_R2"] = {n: [dict(layer=l,
                                z_only=ridge_r2(feats[n]["rd"][1][:, l], G["rd"], feats[n]["te"][1][:, l], G["te"]),
                                z_plus_state=ridge_r2(np.c_[feats[n]["rd"][1][:, l], S["rd"]], G["rd"],
                                                      np.c_[feats[n]["te"][1][:, l], S["te"]], G["te"]))
                           for l in range(feats[n]["rd"][1].shape[1])] for n in ("jepa", "random")}
    # privileged tracked-position gravity fit: noiseless identifiability ceiling
    cp, _ = te.clips(a.c, [1]); tc = sim.frame_tau(np.arange(sim.ANCHOR - a.c + 1, sim.ANCHOR + 1))
    dev = cp - te.expand(te.p0)[:, None] - te.expand(te.v0)[:, None] * tc[None, :, None]
    res["tracked_position_gravity_MAE"] = float(np.abs(-2 * (dev[..., 2] * tc ** 2).sum(1) / (tc ** 4).sum() - G["te"]).mean())

    # CAUSAL INTERVENTION: matched-state history swap. Replace z_A by z_B taken from the SAME anchor under law B
    # (on-manifold: that exact z occurs in the data). swap_effect = <f(s,z_B)-f(s,z_A), Y_B-Y_A> / |Y_B-Y_A|^2:
    # 1 = prediction moves fully to the donor law's future, 0 = no movement.
    n = te.n_groups
    def swap(zfn):
        M_ = np.zeros((K, K))
        for A in range(K):
            for B in range(K):
                if A == B: continue
                ia, ib = np.arange(n) * K + A, np.arange(n) * K + B
                Xs = Xte[ia].copy(); Xs[:, 6:] = zfn(ia, ib)
                d_pred = pr(Xs) - pr(Xte[ia]); d_true = (fut["te"][ib] - fut["te"][ia]).reshape(n, -1)
                M_[A, B] = ((d_pred * d_true).sum(1) / (d_true ** 2).sum(1)).mean()
        return dict(mean=float(M_[~np.eye(K, dtype=bool)].mean()), matrix=M_.tolist())
    res["swap_matched_donor"] = swap(lambda ia, ib: zte[ib])
    res["swap_mean_ablation"] = swap(lambda ia, ib: np.tile(zte.mean(0), (len(ib), 1)))
    res["swap_other_anchor_donor"] = swap(lambda ia, ib: zte[np.roll(ib, 1)])     # off-matched donor: should NOT track law B
    res["runtime_s"] = time.time() - t0
    json.dump(res, open(f"{a.out}/results.json", "w"), indent=2, default=str)
    keys = ("mean_losses", "ARS_jepa", "ARS_random_enc", "ARS_shuffled_history", "swap_matched_donor",
            "swap_mean_ablation", "swap_other_anchor_donor", "tracked_position_gravity_MAE", "runtime_s")
    print(json.dumps({k: res[k] for k in keys}, indent=1, default=str))


if __name__ == "__main__":
    main()
