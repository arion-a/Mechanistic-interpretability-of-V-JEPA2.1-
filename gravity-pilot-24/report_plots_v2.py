"""Plots + raw plot data for the gravity pilot on DATASET v2.0 (48 clips, 12 start positions). Reads ONLY results/predictions.csv and results/summary.json.
Descriptive only: no uncertainty was estimated, so no error bars / intervals are drawn."""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e1"
C = {"full": "#2a78d6", "projected": "#eb6834", "cv": "#1baf7a", "prior": "#eda100"}   # validated slots 1-4 (series colour = method, NOT ball colour)
LAB = {"full": "Full (768-D)", "projected": "Projected (16-D)", "cv": "Constant-velocity baseline", "prior": "Training-prior baseline"}
NOTE = "Descriptive pilot (dataset v2.0): 48 clips, 12 start-position groups; no uncertainty estimated. Series colours denote method, not ball colour."


def _style(ax):
    ax.set_facecolor(SURF); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9); ax.title.set_color(INK); ax.xaxis.label.set_color(INK2); ax.yaxis.label.set_color(INK2)


def _fig(w=9.5, h=5.2):
    f = plt.figure(figsize=(w, h), facecolor=SURF); return f


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def make_all(res: Path):
    S = json.loads((res / "summary.json").read_text())
    rows = list(csv.DictReader(open(res / "predictions.csv")))
    pd = res / "plot_data"; pd.mkdir(exist_ok=True)
    keys = ["full", "projected", "cv", "prior"]

    # 1 six-group position RMSE (+ pooled shown apart, computed from pooled squared errors, not an average of groups)
    fig = _fig(13.5, 5.4); ax = fig.add_axes([0.06, 0.3, 0.93, 0.6]); _style(ax)
    NP = 12
    groups = [str(i) for i in range(NP)]
    xpos = list(range(NP)) + [NP + 0.9]
    data = {k: [S["group_position_rmse_m"][k][g] for g in groups] + [S["pooled_position_rmse_m"][k]] for k in keys}
    bw = 0.17
    for j, k in enumerate(keys):
        ax.bar(np.array(xpos) + (j - 1.5) * (bw + 0.02), data[k], bw, color=C[k], label=LAB[k], zorder=3)
    for i, x in enumerate(xpos):
        v = data["full"][i]; ax.text(x - 1.5 * (bw + 0.02), v + 0.4, f"{v:.1f}", ha="center", fontsize=6.5, color=INK)
        v = data["projected"][i]; ax.text(x - 0.5 * (bw + 0.02), v + 0.4, f"{v:.1f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(xpos); ax.set_xticklabels([f"held-out\nstart {g}" for g in groups] + ["pooled\n(48 clips)"])
    ax.set_ylabel("Euclidean position RMSE at t = 4 s (m)"); ax.set_title("Future-position error by held-out start position", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), labelcolor=INK2)
    fig.text(0.08, 0.045, NOTE, fontsize=7, color=INK2)
    fig.text(0.08, 0.015, "Value labels shown for full and projected only; all values in plot_data/plot1_group_rmse.csv.", fontsize=7, color=INK2)
    fig.savefig(res / "plot1_group_position_rmse.png", dpi=170); plt.close(fig)
    _write(pd / "plot1_group_rmse.csv", ["group"] + keys, [[g if g != "pooled" else g] + [data[k][i] for k in keys] for i, g in enumerate(groups + ["pooled"])])

    # 2 colour correct counts per held-out group, full vs projected
    fig = _fig(12.5, 4.8); ax = fig.add_axes([0.06, 0.22, 0.93, 0.66]); _style(ax)
    acc = S["color"]["group_accuracy"]; xs = np.arange(NP)
    for j, k in enumerate(["full", "projected"]):
        vals = [round(acc[k][g] * 4) for g in groups]
        ax.bar(xs + (j - 0.5) * 0.3, vals, 0.28, color=C[k], label=f"{LAB[k]}: {S['color']['correct_of_48'][k]}/48 correct", zorder=3)
        for x, v in zip(xs + (j - 0.5) * 0.3, vals): ax.text(x, v + 0.07, str(v), ha="center", fontsize=9, color=INK)
    ax.axhline(2, color=INK2, lw=0.8, ls=(0, (4, 3))); ax.set_xlim(-0.6, NP + 0.4); ax.text(NP - 0.38, 2.0, "chance\n(2 of 4)", fontsize=8, color=INK2, ha="left", va="center")
    ax.set_ylim(0, 4.6); ax.set_yticks(range(5)); ax.set_xticks(xs); ax.set_xticklabels([f"held-out\nstart {g}" for g in groups])
    ax.set_ylabel("Colour clips correct (of 4)"); ax.set_title("Orange-vs-blue ridge classifier, held-out start positions", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, labelcolor=INK2)
    fig.text(0.08, 0.02, NOTE, fontsize=7, color=INK2)
    fig.savefig(res / "plot2_color_correct.png", dpi=170); plt.close(fig)
    _write(pd / "plot2_color_correct.csv", ["group", "full_correct_of_4", "projected_correct_of_4"], [[g, round(acc["full"][g] * 4), round(acc["projected"][g] * 4)] for g in groups])

    # 3 paired true vs predicted future positions (world coordinates), by gravity
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 5.4), facecolor=SURF, sharey=False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.88, wspace=0.18)
    for ax, g in zip(axs, ("9.8", "4.9")):
        _style(ax); ax.grid(axis="x", color=GRID, lw=0.6)
        R = [r for r in rows if r["gravity"] == g]
        for r in R:
            t = np.array([float(r["true_x"]), float(r["true_y"])])
            for k, m in (("full", "o"), ("projected", "s")):
                p = np.array([float(r[f"{k}_x"]), float(r[f"{k}_y"])])
                ax.plot([t[0], p[0]], [t[1], p[1]], color=C[k], lw=0.9, alpha=0.8, zorder=2)
                ax.scatter(*p, s=46, marker=m, color=C[k], edgecolor=SURF, lw=1.2, zorder=3)
        T = sorted({(float(r["true_x"]), float(r["true_y"])) for r in R})
        ax.scatter([t[0] for t in T], [t[1] for t in T], s=70, marker="x", color=INK, lw=1.8, zorder=4)
        ax.set_title(f"g = {g} m/s²", loc="left", fontsize=11); ax.set_xlabel("x at 4 s (m)")
    axs[0].set_ylabel("y at 4 s (m)")
    h = [plt.Line2D([], [], marker="x", color=INK, ls="", ms=8, label="True position (simulator)"),
         plt.Line2D([], [], marker="o", color=C["full"], ls="", ms=7, label=LAB["full"] + " out-of-fold prediction"),
         plt.Line2D([], [], marker="s", color=C["projected"], ls="", ms=7, label=LAB["projected"] + " out-of-fold prediction")]
    fig.legend(handles=h, frameon=False, fontsize=8.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.06), labelcolor=INK2)
    fig.suptitle("True vs predicted ball position at t = 4 s (lines join each prediction to its true position)", x=0.07, ha="left", fontsize=12, color=INK)
    fig.text(0.07, 0.015, NOTE, fontsize=7, color=INK2)
    fig.savefig(res / "plot3_true_vs_predicted_positions.png", dpi=170); plt.close(fig)
    _write(pd / "plot3_positions.csv", ["clip_id", "gravity", "color", "true_x", "true_y", "full_x", "full_y", "projected_x", "projected_y"],
           [[r[c] for c in ("clip_id", "gravity", "color", "true_x", "true_y", "full_x", "full_y", "projected_x", "projected_y")] for r in rows])

    # 4 feature payload with fixed overhead clearly separated
    st = S["storage_bytes"]
    fig = _fig(8.6, 5.0); ax = fig.add_axes([0.12, 0.22, 0.85, 0.66]); _style(ax)
    ax.bar([0], [st["full_n"]], 0.5, color=C["full"], zorder=3)
    ax.bar([1], [st["projected_n"]], 0.5, color=C["projected"], zorder=3)
    ax.bar([1], [st["projection_matrix_overhead"]], 0.5, bottom=[st["projected_n"]], color=SURF, edgecolor=C["projected"], hatch="////", lw=1.2, zorder=3)
    ax.text(0, st["full_n"] + 1200, f"{st['full_n']:,} B\n(48 × {st['full_per_clip']:,} B)", ha="center", fontsize=9, color=INK)
    ax.text(1, st["projected_n"] + st["projection_matrix_overhead"] + 1200, f"{st['projected_dataset_incl_matrix']:,} B incl. matrix", ha="center", fontsize=9, color=INK)
    ax.text(1.32, 6500, f"features {st['projected_n']:,} B\n(48 × {st['projected_per_clip']} B)", ha="left", va="center", fontsize=8.5, color=INK)
    ax.text(1.32, st["projected_n"] + st["projection_matrix_overhead"] / 2, f"fixed overhead: projection matrix\n{st['projection_matrix_overhead']:,} B (float32, 16×768)", ha="left", va="center", fontsize=8.5, color=INK2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Full embeddings", "Projected embeddings"]); ax.set_xlim(-0.5, 2.6); ax.set_ylim(0, st["full_n"] * 1.15)
    ax.set_ylabel("Stored float32 bytes (48 clips)")
    ax.set_title(f"Feature payload: {st['reduction_factor']:.0f}× per-example reduction; overhead shown separately", loc="left", fontsize=12)
    fig.text(0.12, 0.02, "Payload accounting only, not Shannon information. Encoder, reader and provenance storage excluded. Raw data: plot_data/plot4_payload.csv", fontsize=7, color=INK2)
    fig.savefig(res / "plot4_feature_payload.png", dpi=170); plt.close(fig)
    _write(pd / "plot4_payload.csv", ["item", "bytes"], [["full_per_clip", st["full_per_clip"]], ["projected_per_clip", st["projected_per_clip"]], ["full_n", st["full_n"]],
           ["projected_n_features", st["projected_n"]], ["projection_matrix_overhead", st["projection_matrix_overhead"]], ["projected_dataset_incl_matrix", st["projected_dataset_incl_matrix"]]])
