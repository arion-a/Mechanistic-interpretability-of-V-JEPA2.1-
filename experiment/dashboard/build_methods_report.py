"""Build the companion methods/math report: every transformation from raw
simulator state to each figure in pilot_dashboard.html, with real worked
numbers from this pilot's own data (not illustrative/fake numbers).

Uses a plain string template (not an f-string) because the page is full of
literal LaTeX braces (\\frac{a}{b}) that would collide with f-string
interpolation; values are substituted via distinctive %%TOKEN%% placeholders.
"""
from __future__ import annotations

import json
import os

import numpy as np

FACTOR_ORDER = ["gravity", "restitution", "friction", "velocity"]


def fmt_vec(v: np.ndarray, n: int = 5) -> str:
    return "[" + ", ".join(f"{x:.4f}" for x in v[:n]) + ", \\ldots]"


def fmt_list(v: list, n: int = 10, prec: int = 4) -> str:
    return "[" + ", ".join(f"{x:.{prec}f}" for x in v[:n]) + ("" if len(v) <= n else ", \\ldots") + "]"


def build_factor_table_rows(analysis: dict) -> str:
    rows = []
    for factor in FACTOR_ORDER:
        if factor not in analysis["per_factor"]:
            continue
        pf = analysis["per_factor"][factor]
        ret = pf["loWO_rank_retention"]
        sv = ", ".join(f"{s:.2f}" for s in pf["singular_values"][:5])
        rows.append(
            f'<tr><td>{factor}</td><td>{pf["n_pairs"]}</td>'
            f'<td class="mono">{sv}, &hellip;</td>'
            f'<td>{pf["r90"]}</td><td>{pf["mean_vector_norm"]:.3f}</td>'
            f'<td>{pf["mean_shift_energy_fraction"]:.4f}</td>'
            f'<td>{ret["mean_retention"]:.4f}</td></tr>'
        )
    return "\n".join(rows)


def build_specificity_table(spec: dict) -> str:
    factors = [f for f in FACTOR_ORDER if f in spec]
    header = "<tr><th></th>" + "".join(f"<th>{f}</th>" for f in factors) + "</tr>"
    rows = [header]
    for row_f in factors:
        cells = "".join(f'<td class="mono">{spec[col_f][row_f]:.4f}</td>' for col_f in factors)
        rows.append(f"<tr><th>{row_f}</th>{cells}</tr>")
    return "\n".join(rows)


TEMPLATE = r"""<title>Intervention Geometry Methods</title>
<style>
%%KATEX_CSS%%
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.11/katex.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.11/contrib/auto-render.min.js"></script>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e;
  --muted: #898781; --gridline: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --accent: #2a78d6; --accent-bg: #eaf1fc;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7;
    --muted: #898781; --gridline: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --accent: #3987e5; --accent-bg: #142236;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7;
  --muted: #898781; --gridline: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --accent: #3987e5; --accent-bg: #142236;
}
* { box-sizing: border-box; }
body { background: var(--page); color: var(--text-primary); font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 860px; margin: 0 auto; padding: 32px 20px 80px; }
h1 { font-size: 27px; margin: 0 0 6px; letter-spacing: -0.01em; text-wrap: balance; }
.subtitle { color: var(--text-secondary); font-size: 15px; margin: 0 0 8px; }
.crossref { font-size: 13px; color: var(--muted); margin: 0 0 28px; }
.crossref a { color: var(--accent); text-decoration: none; }
h2 { font-size: 19px; margin: 40px 0 4px; letter-spacing: -0.005em; }
.stage-num { color: var(--accent); font-variant-numeric: tabular-nums; margin-right: 8px; }
h3 { font-size: 14px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-secondary);
  margin: 20px 0 8px; font-weight: 700; }
p { line-height: 1.65; font-size: 15px; }
.card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 22px 24px; margin: 10px 0 24px;
}
.worked {
  background: var(--accent-bg); border-left: 3px solid var(--accent); border-radius: 0 8px 8px 0;
  padding: 14px 18px; margin: 14px 0; font-size: 14px;
}
.worked .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; font-weight: 700;
  color: var(--accent); margin-bottom: 6px; }
.mono { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12.5px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; margin: 12px 0; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--gridline); }
th { color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 10.5px; letter-spacing: 0.03em; }
td.mono, th.mono { font-variant-numeric: tabular-nums; }
.overflow-x { overflow-x: auto; }
.pipeline { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 20px 0; font-size: 12.5px; }
.pipeline .step { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 8px 12px; white-space: nowrap; }
.pipeline .arrow { color: var(--muted); }
.katex-display { margin: 0.8em 0; overflow-x: auto; overflow-y: hidden; }
code { background: var(--gridline); padding: 1px 5px; border-radius: 4px; font-size: 12.5px; }
footer { color: var(--muted); font-size: 12.5px; margin-top: 40px; line-height: 1.6; }
footer a { color: var(--text-secondary); }
</style>
<div class="wrap">
  <h1>Intervention Geometry Methods</h1>
  <p class="subtitle">Every transformation from simulator state to each number and figure in the pilot dashboard, worked through with this run's own real values.</p>
  <p class="crossref">Companion to the results dashboard &middot; source: <code>experiment/analysis/svd_analysis.py</code> and <code>experiment/encode/encode_pilot.py</code> in the repository.</p>

  <div class="pipeline">
    <div class="step">1&nbsp;&middot;&nbsp;simulate state</div><div class="arrow">&rarr;</div>
    <div class="step">2&nbsp;&middot;&nbsp;render frame</div><div class="arrow">&rarr;</div>
    <div class="step">3&nbsp;&middot;&nbsp;sample 64 frames</div><div class="arrow">&rarr;</div>
    <div class="step">4&nbsp;&middot;&nbsp;encode &amp; pool &rarr; z</div><div class="arrow">&rarr;</div>
    <div class="step">5&nbsp;&middot;&nbsp;&Delta;z per pair</div><div class="arrow">&rarr;</div>
    <div class="step">6&nbsp;&middot;&nbsp;stack per factor</div><div class="arrow">&rarr;</div>
    <div class="step">7&nbsp;&middot;&nbsp;SVD</div><div class="arrow">&rarr;</div>
    <div class="step">8&nbsp;&middot;&nbsp;spectrum / specificity / retention</div>
  </div>

  <h2><span class="stage-num">01</span>Physical state</h2>
  <div class="card">
    <p>The simulator tracks a 13-number state at every physics step: 3D position, a 4-number orientation
    quaternion, 3D linear velocity, and 3D angular velocity.</p>
    $$S_t = (p_t,\ q_t,\ v_t,\ \omega_t) \in \mathbb{R}^{13}, \qquad p_t \in \mathbb{R}^3,\ q_t \in \mathbb{R}^4,\ v_t,\omega_t \in \mathbb{R}^3$$
    <p>Each base world samples one random anchor state at <span class="mono">t=0</span> (position in a box,
    horizontal speed and heading, vertical velocity), then simulates the baseline physics and every
    intervention from that <em>identical</em> anchor, so only the changed constant can explain any
    downstream difference.</p>
    <div class="worked">
      <div class="label">Worked example &mdash; world000's anchor</div>
      <p class="mono">x&#8320;=%%X0%%, y&#8320;=%%Y0%%, z&#8320;=%%Z0%%, heading=%%HEADING%% rad, speed=%%SPEED%% m/s, v_z&#8320;=%%VZ0%% m/s</p>
      <p>Every one of world000's 9 clips (baseline + 8 interventions) starts from exactly this state.</p>
    </div>
  </div>

  <h2><span class="stage-num">02</span>Camera projection</h2>
  <div class="card">
    <p>A fixed camera (view matrix <span class="mono">V</span>, projection matrix <span class="mono">P</span>)
    maps each 3D point to normalized device coordinates, then to a pixel on the 256&times;256 frame:</p>
    $$\text{ndc} = \frac{(p,1)\, V^\top P^\top}{w}, \qquad \text{px} = \frac{\text{ndc}_x+1}{2}\cdot 256,\quad \text{py} = \frac{1-\text{ndc}_y}{2}\cdot 256$$
    <p>This is the exact projection used both to render every frame and, in the dashboard, to find the
    sphere's true on-screen position for the zoomed sample-frame crops &mdash; not a separate estimate.</p>
  </div>

  <h2><span class="stage-num">03</span>Frame sampling</h2>
  <div class="card">
    <p>Each rendered clip is 180 frames (6&nbsp;seconds at 30&nbsp;fps). The encoder's primary input is 64
    frames spanning the full clip, chosen by uniform index sampling:</p>
    $$i_k = \mathrm{round}\!\left(k \cdot \frac{179}{63}\right), \qquad k = 0, 1, \ldots, 63$$
    <p>so the 64 sampled frames cover the entire 6&nbsp;second event window rather than a shorter clipped
    segment, preserving coarse event timing (e.g. when a bounce happens).</p>
  </div>

  <h2><span class="stage-num">04</span>Encoding one clip into one vector <span class="mono">z</span></h2>
  <div class="card">
    <p>The 64 frames pass through the frozen V-JEPA 2 encoder (ViT-L/16, patch size 16, tubelet size 2,
    hidden width 1024). Its 3D patch embedding divides the 64&times;256&times;256 input into
    <span class="mono">64/2 = 32</span> time-tubelets &times; <span class="mono">(256/16)&sup2; = 256</span>
    spatial patches, giving 8,192 output tokens of width 1,024:</p>
    $$H \in \mathbb{R}^{32 \times 256 \times 1024}, \qquad H_{t,s,:} = \text{encoder output for tubelet } t,\ \text{patch } s$$
    <p>The representation contract averages over the 256 spatial patches within each time-tubelet, then
    concatenates the 32 resulting 1,024-vectors end to end:</p>
    $$z = \bigoplus_{t=1}^{32} \left(\frac{1}{256}\sum_{s=1}^{256} H_{t,s,:}\right) \ \in\ \mathbb{R}^{32 \times 1024 = 32{,}768}$$
    <p>This keeps the model's own 32-step time resolution rather than collapsing the whole clip into one
    blurred average, per the design doc's extraction contract.</p>
    <div class="worked">
      <div class="label">Worked example &mdash; world000_baseline's z (first 5 of 32,768 numbers)</div>
      <p class="mono">z = %%Z_BASE_VEC%%</p>
      <p class="mono">&Vert;z&Vert; = %%Z_BASE_NORM%%</p>
    </div>
  </div>

  <h2><span class="stage-num">05</span>&Delta;z: the effect of one intervention</h2>
  <div class="card">
    <p>For a base world and one physical factor, subtract the baseline's z from the intervention clip's z.
    Both clips share the identical anchor state, seed, appearance and camera &mdash; only the one physics
    constant differs &mdash; so &Delta;z isolates that constant's effect on the representation:</p>
    $$\Delta z = z(\text{intervention clip}) - z(\text{baseline clip}) \ \in\ \mathbb{R}^{32{,}768}$$
    <div class="worked">
      <div class="label">Worked example &mdash; world000, gravity: 9.8 &rarr; 12.25 m/s&sup2;</div>
      <p class="mono">z(baseline)[:5]       = %%Z_BASE_VEC%%</p>
      <p class="mono">z(gravity_high)[:5]   = %%Z_GRAV_VEC%%</p>
      <p class="mono">&Delta;z[:5]              = %%DZ_VEC%%</p>
      <p class="mono">&Vert;&Delta;z&Vert; = %%DZ_NORM%% &nbsp; (versus &Vert;z&Vert; &asymp; %%Z_BASE_NORM%% &mdash; the intervention moves the representation by about %%DZ_FRAC%%% of its own length)</p>
    </div>
  </div>

  <h2><span class="stage-num">06</span>Stacking &Delta;z per factor</h2>
  <div class="card">
    <p>For one factor (e.g. gravity: two levels &times; 5 worlds), stack every &Delta;z as a row of a matrix
    <span class="mono">M<sub>f</sub></span>. This pilot has %%N_GRAVITY%% gravity pairs, %%N_RESTITUTION%%
    restitution, %%N_FRICTION%% friction, %%N_VELOCITY%% velocity (friction and velocity each lost one
    pair to the visibility gate):</p>
    $$M_f \in \mathbb{R}^{n_f \times 32{,}768}, \qquad (M_f)_{i,:} = \Delta z_i$$
  </div>

  <h2><span class="stage-num">07</span>Uncentered SVD &mdash; the spectrum chart</h2>
  <div class="card">
    <p>Singular value decomposition factors <span class="mono">M<sub>f</sub></span> into orthogonal
    directions ranked by how much squared length ("energy") each one explains:</p>
    $$M_f = U \Sigma V^\top, \qquad \Sigma = \mathrm{diag}(\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_{n_f})$$
    $$\text{energy}_i = \sigma_i^2, \qquad \text{cumulative energy}(r) = \frac{\sum_{i=1}^{r}\sigma_i^2}{\sum_{i=1}^{n_f}\sigma_i^2}$$
    $$r_{90} = \min\{\, r : \text{cumulative energy}(r) \ge 0.90 \,\}$$
    <p>This is <em>uncentered</em> SVD &mdash; it decomposes the raw &Delta;z rows through the origin, the
    primary "total effect" geometry the design doc calls for (as opposed to centered SVD, which would
    first subtract the per-factor mean and describe variation <em>around</em> that mean).</p>
    <div class="worked">
      <div class="label">Worked example &mdash; gravity's %%N_GRAVITY%% singular values</div>
      <p class="mono">&sigma; = %%GRAVITY_SV%%</p>
      <p class="mono">cumulative energy = %%GRAVITY_CUMENERGY%%</p>
      <p class="mono">r&#8330;&#8320; = %%GRAVITY_R90%% (out of %%N_GRAVITY%% possible ranks)</p>
      <p>r90 sits almost at the sample count for every factor (see appendix table) &mdash; with this few
      rows in a 32,768-dimensional space, SVD can <em>always</em> reconstruct 90% of the energy in close
      to <span class="mono">n&minus;1</span> directions regardless of any real structure. This is a sample-size
      ceiling, not evidence of a compact code; it only becomes informative once <span class="mono">n</span> is
      much larger than the rank being tested, which needs the confirmatory study's hundreds of pairs.</p>
    </div>
  </div>

  <h2><span class="stage-num">08</span>Mean-shift energy fraction</h2>
  <div class="card">
    <p>How much of a factor's average squared &Delta;z length is explained by one single "always shift this
    way" vector &mdash; the mean of all its &Delta;z &mdash; versus needing a different direction per world:</p>
    $$m_f = \frac{1}{n_f}\sum_{i=1}^{n_f} \Delta z_i, \qquad \text{fraction} = \frac{\Vert m_f \Vert^2}{\frac{1}{n_f}\sum_i \Vert \Delta z_i \Vert^2}$$
    <div class="worked">
      <div class="label">Worked example &mdash; gravity</div>
      <p class="mono">&Vert;m<sub>gravity</sub>&Vert; = %%GRAVITY_MEANNORM%%, fraction = %%GRAVITY_MEANFRAC%%</p>
      <p>Roughly %%GRAVITY_MEANFRAC_PCT%%% of gravity's effect is a single consistent shift; the rest
      varies by world &mdash; the effect is not purely additive.</p>
    </div>
  </div>

  <h2><span class="stage-num">09</span>Leave-one-world-out retention</h2>
  <div class="card">
    <p>Tests whether a direction learned from some worlds predicts the direction of change in a world it
    never saw. For each held-out world <span class="mono">w</span>:</p>
    $$V_{-w} = \text{top-}r\text{ right singular vectors of } M_f \text{ fit on all rows except world } w$$
    $$\text{retention}_i = 1 - \frac{\Vert \Delta z_i - \Delta z_i V_{-w}^\top V_{-w} \Vert^2}{\Vert \Delta z_i \Vert^2}, \qquad \text{for each } \Delta z_i \text{ belonging to world } w$$
    <p>Average every fold's retentions together. Rank is capped at %%RANK_USED%% here because
    leave-one-world-out needs at least <span class="mono">rank+1</span> training rows, and this pilot has
    only %%N_WORLDS%% worlds.</p>
    <div class="worked">
      <div class="label">Worked example &mdash; gravity, rank %%RANK_USED%%</div>
      <p class="mono">per-pair retention (5 held-out worlds &times; 2 doses = 10 values) = %%GRAVITY_PERFOLD%%</p>
      <p class="mono">mean retention = %%GRAVITY_MEANRET%%</p>
      <p><strong>Chance-level comparison:</strong> a random rank-%%RANK_USED%% subspace in a
      %%FEATURE_DIM_PLAIN%%-dimensional space captures, in expectation, exactly
      <span class="mono">rank/dim</span> of a fixed vector's energy by symmetry &mdash; a uniformly random
      unit vector's squared component along any fixed r-dimensional subspace averages to r/d regardless of
      the subspace's orientation. Here that is:</p>
      $$\mathbb{E}[\text{retention}_{\text{random}}] = \frac{r}{d} = \frac{%%RANK_USED%%}{%%FEATURE_DIM%%} \approx %%CHANCE_LEVEL%%$$
      <p>Every factor's observed retention (%%RETENTION_RANGE%%) sits far above this %%CHANCE_LEVEL%% floor
      &mdash; a real sign the embedding responds to these interventions in a structured way &mdash; but the
      values are still small in absolute terms and noisy across only %%N_WORLDS%% folds.</p>
    </div>
  </div>

  <h2><span class="stage-num">10</span>Specificity matrix</h2>
  <div class="card">
    <p>Uses the identical retention formula from stage 09, but fits the basis on one factor's &Delta;z and
    tests it on a <em>different</em> factor's &Delta;z:</p>
    $$\text{specificity}[f_{\text{row}}, f_{\text{col}}] = \text{mean}_i\left(1 - \frac{\Vert \Delta z_i - \Delta z_i V_{f_{\text{row}}}^\top V_{f_{\text{row}}} \Vert^2}{\Vert \Delta z_i \Vert^2}\right),\quad \Delta z_i \in M_{f_{\text{col}}}$$
    <p><strong>Fairness fix:</strong> off-diagonal cells are automatically held-out (the basis never saw the
    other factor's data), but naively fitting <span class="mono">V<sub>f</sub></span> on <em>all</em> of
    factor f's own data and testing on that same data (the diagonal) is in-sample and would trivially look
    better than any off-diagonal cell. The diagonal here instead reuses stage 09's held-out retention, so
    every cell &mdash; diagonal included &mdash; reflects a basis tested on data it never saw.</p>
    <div class="worked">
      <div class="label">Full 4&times;4 result (rows = fitted basis, columns = tested factor)</div>
      <div class="overflow-x"><table>
        %%SPECIFICITY_TABLE%%
      </table></div>
      <p>No row's diagonal cell clearly beats every off-diagonal cell in its row (e.g. friction's basis
      explains more of gravity's held-out variation than gravity's own basis does) &mdash; an honest null
      at this sample size, not evidence against specificity.</p>
    </div>
  </div>

  <h2><span class="stage-num">Appendix</span>All four factors, full numbers</h2>
  <div class="card">
    <div class="overflow-x"><table>
      <thead><tr><th>Factor</th><th>Pairs (n<sub>f</sub>)</th><th>Top-5 &sigma;</th><th>r90</th>
      <th>&Vert;mean &Delta;z&Vert;</th><th>Mean-shift fraction</th><th>LOWO retention</th></tr></thead>
      <tbody>
        %%FACTOR_TABLE%%
      </tbody>
    </table></div>
    <p style="margin-top:14px">Feature dimension d = %%FEATURE_DIM_PLAIN%% = 32 time-tubelets &times; 1,024
    hidden width. Rank used for LOWO/specificity = %%RANK_USED%%. Base worlds = %%N_WORLDS%%. Model:
    <code>%%MODEL_ID%%</code> (weights hash <code>%%WEIGHTS_HASH%%</code>).</p>
  </div>

  <footer>
    Generated by <code>experiment/dashboard/build_methods_report.py</code> from
    <code>experiment/results/pilot_analysis.json</code> and the pilot's saved feature vectors &mdash; every
    number above is read directly from that run's output, not re-derived or illustrative. See the
    companion <code>pilot_dashboard.html</code> for the charts these numbers feed, and
    <code>01_3D_VJEPA_Research_Agenda.docx</code> / <code>02_3D_VJEPA_Mathematical_Framework.docx</code> for
    the full confirmatory-study design these formulas implement a pilot-scale version of.
  </footer>
</div>
<script>
  document.addEventListener("DOMContentLoaded", function() {
    try {
      renderMathInElement(document.body, {
        delimiters: [
          {left: "$$", right: "$$", display: true},
          {left: "$", right: "$", display: false}
        ],
        throwOnError: false
      });
    } catch (e) { console.error("KaTeX render failed", e); }
  });
</script>
"""


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.dirname(base)
    pilot_dir = os.path.join(exp_dir, "data", "pilot")
    results_dir = os.path.join(exp_dir, "results")

    with open(os.path.join(base, "assets", "katex-embedded.css")) as f:
        katex_css = f.read()

    with open(os.path.join(pilot_dir, "manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(results_dir, "pilot_analysis.json")) as f:
        analysis = json.load(f)

    base_row = next(r for r in manifest["rows"] if r["clip_id"] == "world000_baseline")
    z_base = np.load(os.path.join(pilot_dir, "features", "world000_baseline.npy"))
    z_grav = np.load(os.path.join(pilot_dir, "features", "world000_gravity_high.npy"))
    dz = z_grav - z_base

    gravity = analysis["per_factor"]["gravity"]
    rank = analysis["rank_used"]
    d = analysis["feature_dim"]
    chance = rank / d

    all_means = [analysis["per_factor"][f]["loWO_rank_retention"]["mean_retention"] for f in FACTOR_ORDER
                 if f in analysis["per_factor"]]

    values = {
        "X0": f"{base_row['anchor_x0']:.4f}", "Y0": f"{base_row['anchor_y0']:.4f}",
        "Z0": f"{base_row['anchor_z0']:.4f}", "HEADING": f"{base_row['anchor_heading']:.4f}",
        "SPEED": f"{base_row['anchor_speed']:.4f}", "VZ0": f"{base_row['anchor_vz0']:.4f}",

        "Z_BASE_VEC": fmt_vec(z_base), "Z_BASE_NORM": f"{np.linalg.norm(z_base):.4f}",
        "Z_GRAV_VEC": fmt_vec(z_grav),
        "DZ_VEC": fmt_vec(dz), "DZ_NORM": f"{np.linalg.norm(dz):.4f}",
        "DZ_FRAC": f"{100*np.linalg.norm(dz)/np.linalg.norm(z_base):.1f}",

        "N_GRAVITY": str(analysis["per_factor"]["gravity"]["n_pairs"]),
        "N_RESTITUTION": str(analysis["per_factor"]["restitution"]["n_pairs"]),
        "N_FRICTION": str(analysis["per_factor"]["friction"]["n_pairs"]),
        "N_VELOCITY": str(analysis["per_factor"]["velocity"]["n_pairs"]),

        "GRAVITY_SV": fmt_list(gravity["singular_values"], n=10, prec=2),
        "GRAVITY_CUMENERGY": fmt_list(gravity["energy_fraction_cumulative"], n=10, prec=4),
        "GRAVITY_R90": str(gravity["r90"]),
        "GRAVITY_MEANNORM": f"{gravity['mean_vector_norm']:.4f}",
        "GRAVITY_MEANFRAC": f"{gravity['mean_shift_energy_fraction']:.4f}",
        "GRAVITY_MEANFRAC_PCT": f"{100*gravity['mean_shift_energy_fraction']:.1f}",
        "GRAVITY_PERFOLD": fmt_list(gravity["loWO_rank_retention"]["per_fold_retention"], n=10, prec=4),
        "GRAVITY_MEANRET": f"{gravity['loWO_rank_retention']['mean_retention']:.4f}",

        "RANK_USED": str(rank), "FEATURE_DIM": f"{d:,}".replace(",", "{,}"),
        "FEATURE_DIM_PLAIN": f"{d:,}",
        "N_WORLDS": str(analysis["n_worlds"]),
        "CHANCE_LEVEL": f"{chance:.6f}",
        "RETENTION_RANGE": f"{min(all_means):.3f}–{max(all_means):.3f}",
        "MODEL_ID": analysis["model_id"], "WEIGHTS_HASH": analysis["weights_hash"],

        "SPECIFICITY_TABLE": build_specificity_table(analysis["specificity_matrix"]),
        "FACTOR_TABLE": build_factor_table_rows(analysis),
        "KATEX_CSS": katex_css,
    }

    html = TEMPLATE
    for token, val in values.items():
        html = html.replace(f"%%{token}%%", val)

    out_path = os.path.join(base, "methods_report.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
