# Mathematical protocol — 24-video gravity pilot, v1.0 (9 Oct 2026)

> **Transcription of `sources/MATHEMATICAL_PROTOCOL.pdf` (pp. 1–9), made because the authoritative `.md` was not supplied.** The PDF governs on any difference. Pp. 10–13 (coding-agent brief, artifacts, handoff) are in `sources/CODING_AGENT_BRIEF.md`. This is a proposed protocol, not a result.

## 1. Purpose
Test predictive compression in frozen JEPA features. Questions: **Q1** can a linear reader of the full final-layer embedding predict ball position at 4 s from the first 2 s? **Q2** does it generalise to an excluded start position? **Q3** does one fixed 16-D projection stay within 10 % relative RMSE? **Q4** is orange/blue readable from full/projected, and does projection reduce it? **Q5** how do paired colour changes affect position predictions? **Q6** storage reduction and per-group behaviour.
Not estimated: Shannon information loss, abstraction-depth score, meta-learning, unseen-gravity generalisation, new object classes. Exactly 24 clips, one checkpoint, one layer, one seed; no training, layer sweep, PCA, other dims/gravities.

## 2. Dataset
s∈{0..5}, a,c∈{0,1}, N=24. g₀=9.8, g₁=4.9 m/s². c=0 orange (235,100,60), c=1 blue (60,140,235). Background (32,32,32); grid (44,44,44). s=3j+i (i∈{0,1,2}, j∈{0,1}); x₀=−40+40i; y₀=140+20j; vₓ=2, v_y=0, r=5 m. World x∈[−100,100], y∈[0,200] m; fixed orthographic camera. 384×384 RGB, 30 fps, 151 frames (t=k/30, k=0..150; 5 s simulated, 5.0333 s playback). Lossless FFV1 MKV; names `sNN_g98|g49_orange|blue.mkv`.

## 3. Physics, mapping, target
x(t)=x₀+2t, y(t)=y₀−½gt² (float64). u=1.92(x+100), v=1.92(200−y), r_px=9.6; grid lines at pixels 0,38,…,380; filled Pillow ellipse bbox (u±9.6, v±9.6), no antialiasing. Centres in x∈[−40,50], y∈[17.5,160].
Encoder frames k_l=4l, t_l=2l/15, l=0..15 (indices 0,4,…,60; t=0..2 s). Frames 61–150 never enter the encoder, preprocessing statistics or representation selection.
Target p=(x₀+8, y₀−8g) m at t=4 s (frame 120). Colour, gravity, position, filename, scene ID, timestamps and simulator parameters are never features.

## 4. Representation and projection
V-JEPA 2.1 ViT-B/16 EMA encoder, revision `204698b45b3712590f06245fbfba32d3be539812`, checkpoint `vjepa2_1_vitb_dist_vitG_384.pt`, strict load; no substitutes. Normalise (I/255−μ_c)/σ_c with μ=(0.485,0.456,0.406), σ=(0.229,0.224,0.225). Input [1,3,16,384,384], float32, frozen, no crop/resize/augmentation/predictor. Final post-LayerNorm tokens, hierarchical output off: T∈ℝ^{4608×768} (8·24·24; assert). z = mean over tokens (raw, unnormalised, float32).
R∈ℝ^{16×768}, R_ij=ξ_ij/√16, ξ∈{−1,+1}: `Generator(PCG64(20261009))`, integers 0/1 row-major (16,768), 2b−1, /4; float32; record SHA-256. u=Rz. No L2 normalisation, no PCA.

## 5. Folds and scaling
U_f={(s,a,c): s=f}, T_f the rest; |T_f|=20, |U_f|=4, f=0..5. One out-of-fold prediction per clip. Per representation r∈{z,u}: r̄_f = mean over T_f; s_f = √(mean‖r_i−r̄_f‖²) (single scalar); a_i=(r_i−r̄_f)/s_f. If s_f≤1e-12 stop and report degenerate. λ=0.01 fixed for both representations and tasks; no nested selection. Save means, scales, coefficients, split IDs. Clips/folds are not independent trials.

## 6. Readers
Position targets p/100 (2 cols); colour targets −1 orange, +1 blue. Ȳ=mean Y; Y_c=Y−1Ȳ;
Ŵ=argmin (1/n)‖AW−Y_c‖²_F+λ‖W‖²_F = Aᵀ(AAᵀ+nλI_n)⁻¹Y_c, n=20, λ=0.01 (linear solve, no explicit inverse; unpenalised intercept via +Ȳ).
Ŷ_i=a_iŴ+Ȳ; p̂_i=100Ŷ_i; ĉ_i=1[Ŷ_i>0] (0→orange). No clipping, no probabilities/cross-entropy/MI. Separate position and colour fits. Reader arithmetic in float64.

## 7. Position metrics and baselines
e_i=‖p̂_i−p_i‖₂; E_r=√(mean e_i²) over 24 (Euclidean). Also x- and y-RMSE, six group RMSEs, per-gravity RMSEs, all 24 errors. Never average group RMSEs as "overall".
**CV baseline (oracle observation):** t_a=56/30, t_b=2, Δ=2/15, v̂=(p(t_b)−p(t_a))/Δ, p̂_CV(4)=p(t_b)+2v̂; error (32/15)g = 20.9066667 m (g=9.8), 10.4533333 m (g=4.9); E_CV=√((20.9066667²+10.4533333²)/2)=16.5282 m.
G_r=1−E_r/E_CV; ρ=E_proj/E_full, pass ρ≤1.10; if E_full≤1e-9 m ρ undefined — report absolute change.
**Training-prior baseline:** d̄_f=mean_{T_f}(p(4)−p(0)) = (8,−58.8) m; p̂_prior=p(0)+d̄_f; E_prior=19.6 m (both gravities). Both baselines use exact simulator centres; neither reads held-out continuation. Learned readers must each beat min(E_CV,E_prior), plus ρ≤1.10.

## 8. Colour, pairing, storage
A_r = (1/24)Σ1[ĉ=c]; ΔA=A_full−A_proj. Report correct/24, accuracy, per-colour recall, confusion matrix, six group accuracies. Reduced-readability marker: ΔA≥3/24=0.125 with A_full>0.5 (descriptive, not significance).
J_r=√((1/12)Σ_{s=0..5}Σ_{a=0,1}‖p̂_{s,a,1}−p̂_{s,a,0}‖²) (colour sensitivity of position predictions; descriptive; J=0 for a constant predictor).
Storage: full 768·32 = 24,576 bit = 3,072 B; projected 16·32 = 512 bit = 64 B; ratio 48×; 1−16/768=97.9167 % (payload, not Shannon information). 24 clips: 73,728 B vs 1,536 B. Projection matrix 16·768·4 = 49,152 B → projected dataset 50,688 B incl. matrix. Encoder/reader/provenance excluded.

## 9. Decision rules
**Primary:** E_full, E_proj < min(E_CV,E_prior) **and** ρ≤1.10 → "the chosen 16-D projection retained useful position prediction within tolerance on these held-out groups" (descriptive).
**Selective accessibility:** A_full>0.5 and ΔA≥0.125 → position retained while colour less readable; if both readable → compact accessibility without selective loss; poor full prediction → no conclusion that compression is effective.
**Breadth:** b_r=(1/6)Σ_f 1[E_{r,f}<min(E_CV,f,E_prior,f)], with all six RMSEs and per-fold ρ. No pooled p-values or binomial CIs; null/mixed results are valid. Paired sensitivity and per-coordinate errors are diagnostics only.
