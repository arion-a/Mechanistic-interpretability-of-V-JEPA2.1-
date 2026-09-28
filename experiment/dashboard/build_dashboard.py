"""Build a self-contained HTML dashboard from the pilot analysis results.

Reads data/pilot/manifest.json + results/pilot_analysis.json (+ a couple of
real rendered frames for grounding) and emits a static HTML report. No chart
library: axes/lines/bars/heatmap are plain inline SVG computed here, sized to
the real data so labels always match ticks the chart reaches.
"""
from __future__ import annotations

import base64
import io
import json
import os

import numpy as np
from PIL import Image

# Reference categorical palette (dataviz skill), fixed hue order, factor-slot mapping
FACTOR_COLORS = {
    "gravity": {"light": "#2a78d6", "dark": "#3987e5"},      # slot 1 blue
    "restitution": {"light": "#eb6834", "dark": "#d95926"},  # slot 2 orange
    "friction": {"light": "#1baf7a", "dark": "#199e70"},     # slot 3 aqua
    "velocity": {"light": "#eda100", "dark": "#c98500"},     # slot 4 yellow
}
FACTOR_ORDER = ["gravity", "restitution", "friction", "velocity"]


def frame_to_data_uri(frame: np.ndarray) -> str:
    img = Image.fromarray(frame)
    img = img.resize((96, 96), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_frame_strip(pilot_dir: str, manifest: dict, clip_ids: list[str], n_thumbs: int = 5) -> dict:
    strips = {}
    by_id = {r["clip_id"]: r for r in manifest["rows"]}
    for clip_id in clip_ids:
        row = by_id.get(clip_id)
        if not row or not row.get("frames_path"):
            continue
        frames = np.load(os.path.join(pilot_dir, row["frames_path"]))
        idx = np.round(np.linspace(0, frames.shape[0] - 1, n_thumbs)).astype(int)
        strips[clip_id] = [frame_to_data_uri(frames[i]) for i in idx]
    return strips


def spectrum_svg(analysis: dict, width=560, height=280) -> str:
    pad_l, pad_r, pad_t, pad_b = 46, 16, 16, 34
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b

    max_len = max(len(v["energy_fraction_cumulative"]) for v in analysis["per_factor"].values())

    def x(i, n):
        return pad_l + (i / max(1, n - 1)) * plot_w

    def y(v):
        return pad_t + (1 - v) * plot_h

    gridlines = "".join(
        f'<line x1="{pad_l}" y1="{y(g):.1f}" x2="{pad_l+plot_w}" y2="{y(g):.1f}" '
        f'class="gridline"/><text x="{pad_l-8}" y="{y(g)+4:.1f}" class="tick" text-anchor="end">{g:.1f}</text>'
        for g in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
    )

    paths = []
    legend = []
    for fi, factor in enumerate(FACTOR_ORDER):
        if factor not in analysis["per_factor"]:
            continue
        cum = analysis["per_factor"][factor]["energy_fraction_cumulative"]
        n = len(cum)
        pts = " ".join(f"{x(i,n):.1f},{y(v):.1f}" for i, v in enumerate(cum))
        color_var = f"var(--series-{factor})"
        paths.append(f'<polyline points="{pts}" fill="none" stroke="{color_var}" stroke-width="2.5" '
                     f'stroke-linecap="round" stroke-linejoin="round">'
                     f'<title>{factor}: r90={analysis["per_factor"][factor]["r90"]}</title></polyline>')
        last_x, last_y = x(n - 1, n), y(cum[-1])
        paths.append(f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="3.5" fill="{color_var}"/>')
        legend.append(f'<span class="legend-item"><i style="background:{color_var}"></i>{factor}</span>')

    axis_labels = (
        f'<text x="{pad_l+plot_w/2:.1f}" y="{height-6}" class="axis-label" text-anchor="middle">singular value rank</text>'
        f'<text x="14" y="{pad_t+plot_h/2:.1f}" class="axis-label" text-anchor="middle" '
        f'transform="rotate(-90 14 {pad_t+plot_h/2:.1f})">cumulative energy</text>'
    )
    x_ticks = "".join(
        f'<text x="{x(i, max_len):.1f}" y="{pad_t+plot_h+18}" class="tick" text-anchor="middle">{i+1}</text>'
        for i in range(max_len)
    )

    return (f'<svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" '
            f'aria-label="Cumulative SVD energy retained by rank, per physical factor">'
            f'{gridlines}{"".join(paths)}{axis_labels}{x_ticks}'
            f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{pad_l+plot_w}" y2="{pad_t+plot_h}" class="axis"/>'
            f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}" class="axis"/>'
            f'</svg>'
            f'<div class="legend">{"".join(legend)}</div>')


def heatmap_svg(spec_matrix: dict, width=420, height=380) -> str:
    factors = [f for f in FACTOR_ORDER if f in spec_matrix]
    n = len(factors)
    pad_l, pad_t = 96, 26
    cell = min((width - pad_l - 16) // max(n, 1), (height - pad_t - 60) // max(n, 1))
    grid_w = cell * n

    vals = [spec_matrix[c][r] for c in factors for r in factors]
    vmin, vmax = min(vals), max(vals)

    def color_for(v):
        # sequential blue ramp, light->dark (dataviz skill: sequential = one hue)
        t = 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)
        steps = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
        idx = min(int(t * (len(steps) - 1)), len(steps) - 1)
        return steps[idx]

    cells = []
    for ri, row_factor in enumerate(factors):
        for ci, col_factor in enumerate(factors):
            v = spec_matrix[col_factor][row_factor]
            cx = pad_l + ci * cell
            cy = pad_t + ri * cell
            fill = color_for(v)
            text_color = "#0b0b0b" if v < (vmin + 0.7 * (vmax - vmin)) else "#ffffff"
            diag = " diag" if ri == ci else ""
            cells.append(
                f'<rect x="{cx}" y="{cy}" width="{cell-2}" height="{cell-2}" fill="{fill}" rx="3" class="heat-cell{diag}">'
                f'<title>fitted={row_factor}, tested={col_factor}: retention={v:.3f}</title></rect>'
                f'<text x="{cx+cell/2:.1f}" y="{cy+cell/2+4:.1f}" text-anchor="middle" '
                f'style="fill:{text_color};font-size:12px;font-weight:600">{v:.2f}</text>'
            )
    row_labels = "".join(
        f'<text x="{pad_l-8}" y="{pad_t+ri*cell+cell/2+4:.1f}" text-anchor="end" class="tick">{f}</text>'
        for ri, f in enumerate(factors)
    )
    col_labels = "".join(
        f'<text x="{pad_l+ci*cell+cell/2:.1f}" y="{pad_t-10}" text-anchor="middle" class="tick">{f}</text>'
        for ci, f in enumerate(factors)
    )
    return (f'<svg viewBox="0 0 {width} {pad_t+grid_w+40}" class="chart-svg" role="img" '
            f'aria-label="Specificity matrix: rows are fitted factor bases, columns are tested factors">'
            f'{row_labels}{col_labels}{"".join(cells)}'
            f'<text x="{pad_l+grid_w/2:.1f}" y="{pad_t+grid_w+28}" class="axis-label" text-anchor="middle">tested factor (column)</text>'
            f'</svg>')


def retention_bars_svg(analysis: dict, width=560, height=260) -> str:
    pad_l, pad_r, pad_t, pad_b = 46, 16, 16, 40
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    factors = [f for f in FACTOR_ORDER if f in analysis["per_factor"]]
    n = len(factors)
    band = plot_w / n
    bar_w = band * 0.5

    def y(v):
        v = max(0.0, min(1.0, v))
        return pad_t + (1 - v) * plot_h

    gridlines = "".join(
        f'<line x1="{pad_l}" y1="{y(g):.1f}" x2="{pad_l+plot_w}" y2="{y(g):.1f}" class="gridline"/>'
        f'<text x="{pad_l-8}" y="{y(g)+4:.1f}" class="tick" text-anchor="end">{g:.1f}</text>'
        for g in (0.0, 0.25, 0.5, 0.75, 1.0)
    )

    bars = []
    for fi, factor in enumerate(factors):
        ret = analysis["per_factor"][factor]["loWO_rank_retention"]
        mean_r = ret["mean_retention"]
        cx = pad_l + fi * band + band / 2
        color_var = f"var(--series-{factor})"
        if mean_r is not None:
            bars.append(f'<rect x="{cx-bar_w/2:.1f}" y="{y(mean_r):.1f}" width="{bar_w:.1f}" '
                         f'height="{pad_t+plot_h-y(mean_r):.1f}" fill="{color_var}" rx="3">'
                         f'<title>{factor}: mean retention {mean_r:.3f} (n={ret["n_folds"]} folds)</title></rect>')
            for pf in ret["per_fold_retention"]:
                py = y(pf)
                bars.append(f'<circle cx="{cx+(np.random.default_rng(hash(factor)%2**32).uniform(-8,8)):.1f}" '
                             f'cy="{py:.1f}" r="2.5" class="fold-dot"/>')
        bars.append(f'<text x="{cx:.1f}" y="{pad_t+plot_h+20}" text-anchor="middle" class="tick">{factor}</text>')

    return (f'<svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" '
            f'aria-label="Leave-one-world-out rank retention per physical factor">'
            f'{gridlines}{"".join(bars)}'
            f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{pad_l+plot_w}" y2="{pad_t+plot_h}" class="axis"/>'
            f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}" class="axis"/>'
            f'<text x="{14}" y="{pad_t+plot_h/2:.1f}" class="axis-label" text-anchor="middle" '
            f'transform="rotate(-90 14 {pad_t+plot_h/2:.1f})">held-out retention</text>'
            f'</svg>')


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.dirname(base)
    pilot_dir = os.path.join(exp_dir, "data", "pilot")
    results_dir = os.path.join(exp_dir, "results")

    with open(os.path.join(pilot_dir, "manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(results_dir, "pilot_analysis.json")) as f:
        analysis = json.load(f)

    sample_clips = ["world000_baseline", "world000_gravity_high", "world000_restitution_high", "world000_friction_low"]
    strips = build_frame_strip(pilot_dir, manifest, sample_clips)

    spectrum_html = spectrum_svg(analysis)
    heatmap_html = heatmap_svg(analysis["specificity_matrix"])
    bars_html = retention_bars_svg(analysis)

    frame_strip_html = ""
    for clip_id, thumbs in strips.items():
        imgs = "".join(f'<img src="{t}" width="72" height="72" alt="frame">' for t in thumbs)
        label = clip_id.replace("world000_", "")
        frame_strip_html += f'<div class="strip"><div class="strip-label">{label}</div><div class="strip-frames">{imgs}</div></div>'

    per_factor_rows = ""
    for factor in FACTOR_ORDER:
        if factor not in analysis["per_factor"]:
            continue
        pf = analysis["per_factor"][factor]
        ret = pf["loWO_rank_retention"]
        per_factor_rows += (
            f'<tr><td><span class="dot" style="background:var(--series-{factor})"></span>{factor}</td>'
            f'<td>{pf["n_pairs"]}</td><td>{pf["r90"]}</td>'
            f'<td>{pf["mean_shift_energy_fraction"]:.3f}</td>'
            f'<td>{ret["mean_retention"]:.3f} (n={ret["n_folds"]})</td></tr>'
        )

    html = f"""<title>Intervention Geometry Pilot</title>
<style>
:root {{
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e;
  --muted: #898781; --gridline: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --warn-bg: #fff4e0; --warn-border: #eda100; --warn-text: #6b4a00;
  --series-gravity: {FACTOR_COLORS['gravity']['light']};
  --series-restitution: {FACTOR_COLORS['restitution']['light']};
  --series-friction: {FACTOR_COLORS['friction']['light']};
  --series-velocity: {FACTOR_COLORS['velocity']['light']};
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7;
    --muted: #898781; --gridline: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --warn-bg: #2a2200; --warn-border: #c98500; --warn-text: #f0c96a;
    --series-gravity: {FACTOR_COLORS['gravity']['dark']};
    --series-restitution: {FACTOR_COLORS['restitution']['dark']};
    --series-friction: {FACTOR_COLORS['friction']['dark']};
    --series-velocity: {FACTOR_COLORS['velocity']['dark']};
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7;
  --muted: #898781; --gridline: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --warn-bg: #2a2200; --warn-border: #c98500; --warn-text: #f0c96a;
  --series-gravity: {FACTOR_COLORS['gravity']['dark']};
  --series-restitution: {FACTOR_COLORS['restitution']['dark']};
  --series-friction: {FACTOR_COLORS['friction']['dark']};
  --series-velocity: {FACTOR_COLORS['velocity']['dark']};
}}
* {{ box-sizing: border-box; }}
body {{ background: var(--page); color: var(--text-primary); font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
.wrap {{ max-width: 980px; margin: 0 auto; padding: 32px 20px 64px; }}
h1 {{ font-size: 28px; margin: 0 0 4px; letter-spacing: -0.01em; text-wrap: balance; }}
.subtitle {{ color: var(--text-secondary); font-size: 15px; margin: 0 0 20px; }}
.pilot-banner {{
  background: var(--warn-bg); border: 1px solid var(--warn-border); color: var(--warn-text);
  border-radius: 10px; padding: 14px 16px; font-size: 13.5px; line-height: 1.5; margin-bottom: 28px;
}}
.pilot-banner strong {{ display: inline-block; margin-bottom: 2px; }}
.card {{
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 20px; margin-bottom: 20px;
}}
.card h2 {{ font-size: 15px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-secondary);
  margin: 0 0 4px; font-weight: 600; }}
.card .desc {{ font-size: 13px; color: var(--muted); margin: 0 0 16px; }}
.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
@media (max-width: 760px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
.meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 14px 20px; }}
.meta-item .label {{ font-size: 11.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.03em; }}
.meta-item .value {{ font-size: 16px; font-weight: 600; font-variant-numeric: tabular-nums; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--gridline); font-variant-numeric: tabular-nums; }}
th {{ color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.03em; }}
.dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 7px; }}
.chart-svg {{ width: 100%; height: auto; display: block; }}
.gridline {{ stroke: var(--gridline); stroke-width: 1; }}
.axis {{ stroke: var(--axis); stroke-width: 1; }}
.tick {{ fill: var(--muted); font-size: 11px; }}
.axis-label {{ fill: var(--text-secondary); font-size: 11.5px; }}
.fold-dot {{ fill: var(--muted); opacity: 0.7; }}
.heat-cell.diag {{ stroke: var(--text-primary); stroke-width: 1.5; stroke-opacity: 0.35; }}
.legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 10px; font-size: 12.5px; color: var(--text-secondary); }}
.legend-item {{ display: inline-flex; align-items: center; gap: 6px; }}
.legend-item i {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
.strip {{ margin-bottom: 14px; }}
.strip-label {{ font-size: 12.5px; color: var(--text-secondary); margin-bottom: 6px; font-weight: 600; }}
.strip-frames {{ display: flex; gap: 4px; overflow-x: auto; }}
.strip-frames img {{ border-radius: 4px; border: 1px solid var(--border); flex-shrink: 0; }}
footer {{ color: var(--muted); font-size: 12.5px; margin-top: 32px; line-height: 1.6; }}
footer a {{ color: var(--text-secondary); }}
code {{ background: var(--gridline); padding: 1px 5px; border-radius: 4px; font-size: 12px; }}
</style>
<div class="wrap">
  <h1>Intervention Geometry Pilot</h1>
  <p class="subtitle">V-JEPA 2 embedding response to one-factor physics interventions on a simulated falling/bouncing sphere</p>

  <div class="pilot-banner">
    <strong>⚠ Engineering pilot, not the confirmatory study.</strong>
    {analysis['pilot_scale_caveat']}
    Renderer: PyBullet TinyRenderer (Blender EEVEE unavailable in this sandbox) &mdash;
    sanctioned in the design doc for smoke tests only, not the final 1,200-world split.
    Model: <code>{analysis['model_id']}</code>, a documented substitution for the docx's
    unpublished ViT-B/16 384px checkpoint.
  </div>

  <div class="card">
    <h2>Run metadata</h2>
    <div class="meta-grid">
      <div class="meta-item"><div class="label">Base worlds</div><div class="value">{analysis['n_worlds']}</div></div>
      <div class="meta-item"><div class="label">Accepted clips</div><div class="value">{analysis['n_accepted_clips']}</div></div>
      <div class="meta-item"><div class="label">Feature dim</div><div class="value">{analysis['feature_dim']}</div></div>
      <div class="meta-item"><div class="label">Rank used</div><div class="value">{analysis['rank_used']}</div></div>
      <div class="meta-item"><div class="label">Weights hash</div><div class="value">{analysis['weights_hash']}</div></div>
    </div>
  </div>

  <div class="card">
    <h2>Sample clips (real rendered frames)</h2>
    <p class="desc">5 uniformly spaced frames from world000 across baseline and three interventions &mdash; ground truth for what the encoder actually saw.</p>
    {frame_strip_html}
  </div>

  <div class="grid-2">
    <div class="card">
      <h2>Uncentered SVD spectrum</h2>
      <p class="desc">Cumulative reconstruction energy vs. rank, fit separately per factor on baseline&rarr;intervention &Delta;z.</p>
      {spectrum_html}
    </div>
    <div class="card">
      <h2>Specificity matrix</h2>
      <p class="desc">Row = basis fit to that factor's &Delta;z; column = energy retained on another factor's &Delta;z at equal rank.</p>
      {heatmap_html}
    </div>
  </div>

  <div class="card">
    <h2>Leave-one-world-out rank retention</h2>
    <p class="desc">Fit rank-{analysis['rank_used']} basis on all-but-one world, measure retained energy on the held-out world's &Delta;z. Dots are individual folds (n={analysis['n_worlds']}).</p>
    {bars_html}
  </div>

  <div class="card">
    <h2>Per-factor summary</h2>
    <table>
      <thead><tr><th>Factor</th><th>Pairs</th><th>r90</th><th>Mean-shift energy frac.</th><th>LOWO retention</th></tr></thead>
      <tbody>{per_factor_rows}</tbody>
    </table>
  </div>

  <footer>
    Generated from a {analysis['n_worlds']}-world pilot run on a CPU-only sandbox (no GPU, no Blender).
    Diagonal cells in the specificity matrix are outlined. Numbers here demonstrate the analysis
    pipeline end-to-end; they are not statistically powered evidence for or against the
    low-rank/factor-specificity hypothesis in <code>01_3D_VJEPA_Research_Agenda.docx</code> &mdash;
    that requires the full 1,200-world confirmatory study with Blender rendering and GPU-scale
    V-JEPA inference.
  </footer>
</div>
"""

    out_path = os.path.join(base, "pilot_dashboard.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
