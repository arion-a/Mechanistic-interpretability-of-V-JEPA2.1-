"""SVD / specificity / retention analysis on pilot Delta-z vectors.

Implements the core estimands from 02_3D_VJEPA_Mathematical_Framework.docx:
uncentered SVD on paired intervention differences, r90, rank-8 retention
(here: rank min(8, n_train-1) since the pilot has far fewer worlds than the
confirmatory study), a centered-vs-uncentered energy-fraction check, and a
specificity matrix across physical factors.

PILOT-SCALE CAVEAT: this runs on a handful of base worlds (see
data/pilot/manifest.json), not the confirmatory 1,200-world study. Every
number here is an engineering-pilot demonstration of the analysis code, not
a statistically powered finding. Do not read confidence into small effect
sizes computed from this few worlds.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

PHYSICAL_FACTORS = ["gravity", "restitution", "friction", "velocity"]


def load_features(pilot_dir: str) -> pd.DataFrame:
    feat_dir = os.path.join(pilot_dir, "features")
    with open(os.path.join(feat_dir, "features_manifest.json")) as f:
        meta = json.load(f)
    rows = meta["rows"]
    for r in rows:
        r["feature"] = np.load(os.path.join(feat_dir, r["feature_path"]))
    df = pd.DataFrame(rows)
    return df, meta


def build_delta_z(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Delta-z per factor: rows are (world, intervention) pairs vs. that world's baseline."""
    by_world = {w: g.set_index("intervention") for w, g in df.groupby("world_index")}
    deltas: dict[str, list] = {f: [] for f in PHYSICAL_FACTORS}
    world_ids: dict[str, list] = {f: [] for f in PHYSICAL_FACTORS}

    for world_idx, group in by_world.items():
        if "baseline" not in group.index:
            continue
        z_base = group.loc["baseline", "feature"]
        for intervention in group.index:
            if intervention == "baseline":
                continue
            factor = group.loc[intervention, "factor"]
            if factor not in PHYSICAL_FACTORS:
                continue
            dz = group.loc[intervention, "feature"] - z_base
            deltas[factor].append(dz)
            world_ids[factor].append(world_idx)

    return (
        {f: np.stack(v) for f, v in deltas.items() if v},
        {f: np.array(v) for f, v in world_ids.items() if v},
    )


def uncentered_svd_spectrum(delta_matrix: np.ndarray):
    """SVD of raw (uncentered) rows: minimizes squared reconstruction error through the origin."""
    U, S, Vt = np.linalg.svd(delta_matrix, full_matrices=False)
    energy = S ** 2
    total_energy = energy.sum()
    cumulative = np.cumsum(energy) / total_energy if total_energy > 0 else np.zeros_like(energy)
    r90 = int(np.searchsorted(cumulative, 0.90) + 1) if total_energy > 0 else None
    return {"singular_values": S, "energy_fraction_cumulative": cumulative, "r90": r90, "Vt": Vt}


def centered_vs_uncentered(delta_matrix: np.ndarray) -> dict:
    """Energy fraction explained by the mean shift alone (mathematical-framework doc, section 02)."""
    mean_vec = delta_matrix.mean(axis=0)
    mean_energy = np.sum(mean_vec ** 2)
    per_row_energy = np.mean(np.sum(delta_matrix ** 2, axis=1))
    fraction = float(mean_energy / per_row_energy) if per_row_energy > 0 else None
    return {"mean_shift_energy_fraction": fraction, "mean_vector_norm": float(np.linalg.norm(mean_vec))}


def leave_one_world_out_retention(delta_matrix: np.ndarray, world_ids: np.ndarray, rank: int) -> dict:
    """Fit uncentered rank-r basis on all worlds but one, measure retention on the held-out row(s)."""
    unique_worlds = np.unique(world_ids)
    retentions = []
    for held_out in unique_worlds:
        train_mask = world_ids != held_out
        test_mask = ~train_mask
        if train_mask.sum() < rank + 1 or test_mask.sum() == 0:
            continue
        train = delta_matrix[train_mask]
        test = delta_matrix[test_mask]
        U, S, Vt = np.linalg.svd(train, full_matrices=False)
        r = min(rank, Vt.shape[0])
        basis = Vt[:r]  # [r, d]
        proj = test @ basis.T @ basis  # project onto rank-r subspace
        residual_energy = np.sum((test - proj) ** 2, axis=1)
        total_energy = np.sum(test ** 2, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            retention = 1 - residual_energy / total_energy
        retentions.extend(retention[np.isfinite(retention)].tolist())
    return {
        "rank": rank,
        "mean_retention": float(np.mean(retentions)) if retentions else None,
        "per_fold_retention": retentions,
        "n_folds": len(unique_worlds),
    }


def specificity_matrix(deltas: dict[str, np.ndarray], world_ids: dict[str, np.ndarray],
                        rank: int, diagonal_retention: dict[str, float]) -> pd.DataFrame:
    """Row = fitted factor basis, column = test factor's Delta-z energy retention at equal rank.

    Off-diagonal cells fit the basis on all of row_factor's Delta-z and test on
    col_factor's Delta-z - already a fair held-out comparison since the basis
    never saw col_factor's data. The diagonal (row_factor == col_factor) uses
    the SAME leave-one-world-out retention as the retention chart instead of
    fitting and testing on identical rows, which would trivially look better
    than every off-diagonal cell regardless of any real factor-specific signal.
    """
    factors = list(deltas.keys())
    bases = {}
    for f in factors:
        U, S, Vt = np.linalg.svd(deltas[f], full_matrices=False)
        r = min(rank, Vt.shape[0])
        bases[f] = Vt[:r]

    mat = pd.DataFrame(index=factors, columns=factors, dtype=float)
    for row_factor in factors:
        basis = bases[row_factor]
        for col_factor in factors:
            if row_factor == col_factor:
                mat.loc[row_factor, col_factor] = diagonal_retention[row_factor]
                continue
            test = deltas[col_factor]
            proj = test @ basis.T @ basis
            residual = np.sum((test - proj) ** 2, axis=1)
            total = np.sum(test ** 2, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                retention = 1 - residual / total
            mat.loc[row_factor, col_factor] = float(np.mean(retention[np.isfinite(retention)]))
    return mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", default="../data/pilot")
    ap.add_argument("--out", default="../results")
    args = ap.parse_args()

    pilot_dir = os.path.abspath(args.pilot_dir)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    df, feat_meta = load_features(pilot_dir)
    deltas, world_ids = build_delta_z(df)

    n_worlds = df["world_index"].nunique()
    max_feasible_rank = max(1, n_worlds - 2)  # need >= rank+1 train rows in LOWO CV
    rank = min(8, max_feasible_rank)

    report = {
        "pilot_scale_caveat": (
            f"Computed from {n_worlds} base worlds ({len(df)} accepted clips) - an "
            "engineering-pilot demonstration of the analysis code, not the confirmatory "
            "1,200-world study. Rank capped at {rank} (not the docx's fixed rank-8) "
            "because leave-one-world-out CV needs at least rank+1 training rows."
        ).format(rank=rank),
        "n_worlds": int(n_worlds),
        "n_accepted_clips": int(len(df)),
        "feature_dim": int(feat_meta["rows"][0]["feature_dim"]) if feat_meta["rows"] else None,
        "model_id": feat_meta["model_id"],
        "weights_hash": feat_meta.get("weights_hash", "not_recorded_by_sweeper"),
        "rank_used": rank,
        "per_factor": {},
    }

    for factor, dz in deltas.items():
        spectrum = uncentered_svd_spectrum(dz)
        centered = centered_vs_uncentered(dz)
        retention = leave_one_world_out_retention(dz, world_ids[factor], rank)
        report["per_factor"][factor] = {
            "n_pairs": int(dz.shape[0]),
            "singular_values": spectrum["singular_values"].tolist(),
            "energy_fraction_cumulative": spectrum["energy_fraction_cumulative"].tolist(),
            "r90": spectrum["r90"],
            "mean_shift_energy_fraction": centered["mean_shift_energy_fraction"],
            "mean_vector_norm": centered["mean_vector_norm"],
            "loWO_rank_retention": retention,
        }

    diagonal_retention = {f: report["per_factor"][f]["loWO_rank_retention"]["mean_retention"] for f in deltas}
    spec_mat = specificity_matrix(deltas, world_ids, rank, diagonal_retention)
    report["specificity_matrix"] = spec_mat.to_dict()

    with open(os.path.join(out_dir, "pilot_analysis.json"), "w") as f:
        json.dump(report, f, indent=2)
    spec_mat.to_csv(os.path.join(out_dir, "specificity_matrix.csv"))

    print(json.dumps({k: v for k, v in report.items() if k not in ("per_factor", "specificity_matrix")}, indent=2))
    print("\nSpecificity matrix (row=fitted basis, col=test factor, rank=%d):" % rank)
    print(spec_mat.round(3))
    print(f"\nWrote {out_dir}/pilot_analysis.json and specificity_matrix.csv")


if __name__ == "__main__":
    main()
