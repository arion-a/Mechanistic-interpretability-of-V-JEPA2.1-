# Dataset v1.1 — changes from protocol v1.0 (user-requested, 2026-10-10)

Reason: the v1.0 stimuli were flat dark scenes with no ground, no landing and no bounce. The user asked for a normal-looking video with a coloured scene, ground contact, bouncing and a 3D feel, keeping the same experiment. Chosen options: *shaded scene, same side camera*; *longer clips, bounce after the 4 s target*; restitution 0.6 (my default, confirmed by user "go ahead").

## What changed (stimulus only)
| Item | v1.0 | v1.1 |
|---|---|---|
| Background | flat (32,32,32) + grid | pale sky gradient (196,208,224)→(236,236,228), striped green ground below world y=0 |
| Ball | flat disc | shaded sphere (albedo = protocol colour, upper-left light, specular), hard edge, no antialiasing |
| Shadow | none | hard-edged ellipse on the ground under the ball, fixed size, multiplies ground by 0.70 |
| Clip length | 151 frames (5.0 s simulated) | **271 frames (9.0 s simulated)** at 30 fps |
| Physics | free fall only | free fall until the ball bottom touches y=0 (centre y=5 m), then vertical bounce, restitution e=0.6, constant vx, no friction |
| Vertical view | world y∈[0,200] → rows 0..384 | world y∈[-25,175]; row = 1.92·(175−y). **Same scale, shifted 25 m** so a ground strip is visible (ground surface at row 336) |

## What did NOT change (verified in validation)
* Scenes, positions, gravities, colours (albedo), 24-clip design, folds, readers, λ, seed, projection, decision rules.
* Encoder input: source frame indices 0,4,…,60 (t≤2 s); frames 61–270 never enter the encoder (tested).
* Target: p=(x0+8, y0−8g) at t=4 s (frame 120). Verified equal to the bounce-aware trajectory because the first impact is later: 5.25 s (g=9.8,y0=140), 5.62 s (9.8,160), 7.42 s (4.9,140), 7.95 s (4.9,160).
* Baselines and analytic values: E_CV=16.528 m, E_prior=19.6 m (min rule kept).
* Centre trajectory is identical to the v1.0 formula for the whole first 5 s.

## Consequences to remember
* Frame-count and "5 s" wording in the v1.0 PDF/brief (151 frames, frames 61–150) is superseded by 271 frames / frames 61–270 for this dataset.
* Every pixel differs from v1.0, so embeddings from any v1.0 stimulus must not be reused (none exist).
* The shadow is an additional position cue (horizontal position only; fixed size). Scene background/ground are static.
* Orange/blue stay the exact protocol albedo colours, but visible shaded RGB values vary across the ball surface. Colour is still not luminance-matched.
* The ball's post-landing motion is for viewing only; neither encoder nor target uses it.
* tools/protocol_checks.py still checks the **v1.0** analytic numbers/geometry (flat mapping); pilot.py validate covers v1.1.
* Dataset approval has not been given for either version.
