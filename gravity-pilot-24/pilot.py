#!/usr/bin/env python3
"""24-video gravity pilot, protocol v1.0 — Stage A implementation (generate / validate / gallery).

Commands
  generate         render the 24 lossless FFV1 clips + manifest.json (deterministic; no timestamps)
  validate         check the clips and write dataset_validation.json
  gallery          write derivative viewing copies, contact sheet and gallery.html
  record-approval  record an explicit human dataset approval against the manifest SHA-256
  extract|analyze|report   Stage B — NOT IMPLEMENTED. `extract` refuses without a matching approval receipt.

Nothing here loads a model or touches JEPA features.
"""
import argparse, hashlib, json, math, platform, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import PIL

ROOT = Path(__file__).resolve().parent
DS = ROOT / "dataset"
VID = DS / "videos"
CFG = json.loads((ROOT / "pilot_config.json").read_text())

SIZE, FPS, NFRAMES, PXM, RPX = 384, 30, 151, 1.92, 9.6
GRAV = (9.8, 4.9)
COLORS = (("orange", (235, 100, 60)), ("blue", (60, 140, 235)))
BG, GRID = (32, 32, 32), (44, 44, 44)
GRID_LINES = list(range(0, 381, 38))
OBS_IDX = [4 * l for l in range(16)]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CENTROID_TOL = math.sqrt(2) / 2  # px; rasterised-disc centroid is quantised to half a pixel per axis
PIX_FMT = "bgr0"  # RGB-native lossless pixel format for FFV1 (resolves review item I-07)


# ---------------------------------------------------------------- helpers
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def clip_list():
    out = []
    for s in range(6):
        i, j = s % 3, s // 3
        for a, g in enumerate(GRAV):
            for c, (cn, rgb) in enumerate(COLORS):
                out.append(dict(clip_id=f"s{s:02d}_g{'98' if a == 0 else '49'}_{cn}", scene=s, i=i, j=j,
                                x0=-40 + 40 * i, y0=140 + 20 * j, gravity_idx=a, g=g, color_idx=c, color=cn, rgb=list(rgb)))
    return out


def center(x0, y0, g, t):  # float64 world coordinates, metres
    return x0 + 2.0 * t, y0 - 0.5 * g * t * t


def to_px(x, y):
    return PXM * (x + 100.0), PXM * (200.0 - y)


def render_frame(x0, y0, g, rgb, k, ball=True):
    t = k / FPS
    img = Image.new("RGB", (SIZE, SIZE), BG)
    d = ImageDraw.Draw(img)
    for p in GRID_LINES:  # 1-px lines, drawn under the ball (pins review item I-08)
        d.line([(p, 0), (p, SIZE - 1)], fill=GRID, width=1)
        d.line([(0, p), (SIZE - 1, p)], fill=GRID, width=1)
    if ball:
        u, v = to_px(*center(x0, y0, g, t))
        d.ellipse((u - RPX, v - RPX, u + RPX, v + RPX), fill=tuple(rgb))
    return np.asarray(img, dtype=np.uint8)


def render_clip(c):
    return np.stack([render_frame(c["x0"], c["y0"], c["g"], c["rgb"], k) for k in range(NFRAMES)])


def ffmpeg_encode(frames, path):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE}x{SIZE}", "-r", str(FPS),
           "-i", "-", "-c:v", "ffv1", "-level", "3", "-g", "1", "-pix_fmt", PIX_FMT, "-map_metadata", "-1",
           "-fflags", "+bitexact", "-flags:v", "+bitexact", str(path)]
    subprocess.run(cmd, input=frames.tobytes(), check=True)


def ffmpeg_decode(path):
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       check=True, capture_output=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(-1, SIZE, SIZE, 3)


def ffprobe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name,pix_fmt,width,height,nb_read_packets:packet=pts_time", "-count_packets", "-of", "json", str(path)],
                       check=True, capture_output=True, text=True)
    return json.loads(r.stdout)


def build_encoder_input(frames):
    """Pure input construction for Stage B: only source indices 0,4,...,60 are read.
    frames: [T>=61,H,W,3] uint8 -> float32 [1,3,16,384,384]. No model involved."""
    sel = frames[OBS_IDX].astype(np.float32) / 255.0
    sel = (sel - MEAN) / STD
    return np.ascontiguousarray(sel.transpose(3, 0, 1, 2)[None])


def centroid(frame, rgb):
    m = np.all(frame == np.array(rgb, np.uint8), axis=-1)
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    return xs.mean() + 0.5, ys.mean() + 0.5, int(m.sum())  # +0.5: pixel-centre -> continuous coords


# ---------------------------------------------------------------- generate
def cmd_generate(_):
    VID.mkdir(parents=True, exist_ok=True)
    clips = []
    for c in clip_list():
        frames = render_clip(c)
        p = VID / f"{c['clip_id']}.mkv"
        ffmpeg_encode(frames, p)
        tx, ty = center(c["x0"], c["y0"], c["g"], 4.0)
        clips.append({**c, "filename": f"videos/{c['clip_id']}.mkv", "n_frames": NFRAMES, "fps": FPS, "size": [SIZE, SIZE],
                      "timestamps": "k/30 s, k=0..150", "codec": "ffv1", "container": "matroska", "pix_fmt": PIX_FMT,
                      "target_t4_m": [round(tx, 9), round(ty, 9)], "target_frame": 120,
                      "sha256": sha256_file(p), "raw_rgb_frames_sha256": hashlib.sha256(frames.tobytes()).hexdigest()})
        print("wrote", p.name)
    manifest = {"protocol_version": "1.0", "n_clips": len(clips), "config_sha256": sha256_file(ROOT / "pilot_config.json"),
                "generator_code_sha256": sha256_file(Path(__file__)), "observed_frame_indices": OBS_IDX, "clips": clips}
    mp = DS / "manifest.json"
    mp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    (DS / "provenance_stageA.json").write_text(json.dumps({
        "stage": "A", "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pillow": PIL.__version__,
        "ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0],
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "manifest_sha256": sha256_file(mp), "model_loaded": False, "inference_performed": False}, indent=1) + "\n")
    print("manifest sha256", sha256_file(mp))


# ---------------------------------------------------------------- validate
def cmd_validate(_):
    mp = DS / "manifest.json"
    man = json.loads(mp.read_text())
    res = []

    def chk(name, ok, detail=""):
        res.append({"check": name, "pass": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + name + (f" [{detail}]" if detail else ""))

    clips = man["clips"]
    chk("exactly 24 clips in manifest", len(clips) == 24)
    files = sorted(p.name for p in VID.glob("*.mkv"))
    chk("exactly 24 .mkv files, names match manifest", files == sorted(Path(c["filename"]).name for c in clips) and len(files) == 24, f"{len(files)} files")
    chk("manifest matches current config + generator hashes", man["config_sha256"] == sha256_file(ROOT / "pilot_config.json")
        and man["generator_code_sha256"] == sha256_file(Path(__file__)))
    chk("all file SHA-256 match manifest", all(sha256_file(DS / c["filename"]) == c["sha256"] for c in clips))
    all_err, dec, cent_fail, vis_fail, rt_fail, ts_fail, meta_fail = [], {}, [], [], [], [], []
    for c in clips:
        p = DS / c["filename"]
        info = ffprobe(p)
        st = info["streams"][0]
        ts = [float(x["pts_time"]) for x in info["packets"]]
        if not (st["codec_name"] == "ffv1" and st["pix_fmt"] == PIX_FMT and st["width"] == SIZE and st["height"] == SIZE and int(st["nb_read_packets"]) == NFRAMES):
            meta_fail.append(c["clip_id"])
        if not (len(ts) == NFRAMES and max(abs(t - k / FPS) for k, t in enumerate(ts)) <= 5e-4):
            ts_fail.append(c["clip_id"])
        fr = ffmpeg_decode(p)
        dec[c["clip_id"]] = fr
        if fr.shape != (NFRAMES, SIZE, SIZE, 3) or not np.array_equal(fr, render_clip(c)):
            rt_fail.append(c["clip_id"])
        if hashlib.sha256(fr.tobytes()).hexdigest() != c["raw_rgb_frames_sha256"]:
            rt_fail.append(c["clip_id"] + ":hash")
        worst = 0.0
        errs = []
        for k in range(NFRAMES):
            ct = centroid(fr[k], c["rgb"])
            u, v = to_px(*center(c["x0"], c["y0"], c["g"], k / FPS))
            if ct is None or ct[2] < 0.9 * math.pi * RPX ** 2 or u - RPX < 0 or u + RPX > SIZE or v - RPX < 0 or v + RPX > SIZE:
                vis_fail.append((c["clip_id"], k)); continue
            errs.append((ct[0] - u, ct[1] - v))
        e = np.array(errs)
        worst = float(np.hypot(e[:, 0], e[:, 1]).max())
        all_err.append(e)
        if worst > CENTROID_TOL:
            cent_fail.append((c["clip_id"], round(worst, 3)))
    chk("codec ffv1, pix_fmt bgr0, 384x384, 151 packets", not meta_fail, str(meta_fail))
    chk("151 packets in order; timestamps = k/30 s within 0.5 ms (Matroska 1-ms timebase), last = 5.0 s", not ts_fail, str(ts_fail))
    chk("lossless round-trip: decoded RGB == rendered frames, all 24x151 frames", not rt_fail, str(rt_fail))
    chk("ball fully visible in every frame (>=90% of disc pixels, inside canvas)", not vis_fail, str(vis_fail[:3]))
    E = np.concatenate(all_err)
    mx = float(np.hypot(E[:, 0], E[:, 1]).max())
    chk(f"detected ball centroid within sqrt(2)/2 px (half-pixel per axis, raster quantisation) of analytic trajectory", not cent_fail,
        f"max observed {mx:.3f} px = {mx / PXM:.3f} m; {cent_fail}")
    chk("centroid error is unbiased: |mean dx|,|mean dy| < 0.05 px over all 24x151 frames", abs(E[:, 0].mean()) < 0.05 and abs(E[:, 1].mean()) < 0.05,
        f"mean dx {E[:, 0].mean():.4f}, dy {E[:, 1].mean():.4f} px")
    bad_pair, bad_first = [], []
    for s in range(6):
        for a in range(2):
            ids = [c["clip_id"] for c in clips if c["scene"] == s and c["gravity_idx"] == a]
            cs = {c["clip_id"]: c for c in clips}
            o, b = [i for i in ids if i.endswith("orange")][0], [i for i in ids if i.endswith("blue")][0]
            same = all(math.isclose(centroid(dec[o][k], cs[o]["rgb"])[0], centroid(dec[b][k], cs[b]["rgb"])[0], abs_tol=1e-9) and
                       math.isclose(centroid(dec[o][k], cs[o]["rgb"])[1], centroid(dec[b][k], cs[b]["rgb"])[1], abs_tol=1e-9) for k in range(NFRAMES))
            if not same or cs[o]["target_t4_m"] != cs[b]["target_t4_m"]:
                bad_pair.append((s, a))
        for col in ("orange", "blue"):
            f0 = [dec[f"s{s:02d}_g98_{col}"][0], dec[f"s{s:02d}_g49_{col}"][0]]
            if not np.array_equal(*f0):
                bad_first.append((s, col))
    chk("colour pairs have identical trajectories (centroids) and identical targets", not bad_pair, str(bad_pair))
    chk("first frames identical across gravities within each scene/colour", not bad_first, str(bad_first))
    chk("manifest targets equal analytic (x0+8, y0-8g)", all(
        c["target_t4_m"] == [round(c["x0"] + 8, 9), round(c["y0"] - 8 * c["g"], 9)] for c in clips))
    chk("each of 6 scenes has 4 clips; colours/gravities balanced", all(sum(c["scene"] == s for c in clips) == 4 for s in range(6))
        and sum(c["color"] == "orange" for c in clips) == 12 and sum(c["gravity_idx"] == 0 for c in clips) == 12)
    # encoder-input property (software-only, no model): frames 61..150 cannot affect the tensor
    ref = dec[clips[0]["clip_id"]]
    mod = ref.copy()
    mod[61:] = np.random.default_rng(0).integers(0, 256, size=mod[61:].shape, dtype=np.uint8)
    chk("encoder input tensor unchanged when frames 61-150 are modified", np.array_equal(build_encoder_input(ref), build_encoder_input(mod)))
    t = build_encoder_input(ref)
    chk("encoder input shape [1,3,16,384,384] float32", t.shape == (1, 3, 16, 384, 384) and t.dtype == np.float32)
    mod2 = ref.copy(); mod2[60, 0, 0] ^= 255
    chk("negative control: modifying frame 60 DOES change the tensor", not np.array_equal(build_encoder_input(ref), build_encoder_input(mod2)))
    out = {"manifest_sha256": sha256_file(mp), "all_passed": all(r["pass"] for r in res), "failures": [r for r in res if not r["pass"]], "checks": res}
    (DS / "dataset_validation.json").write_text(json.dumps(out, indent=1) + "\n")
    print("ALL PASSED" if out["all_passed"] else "FAILURES PRESENT")
    sys.exit(0 if out["all_passed"] else 1)


# ---------------------------------------------------------------- gallery (derivative copies only)
def cmd_gallery(_):
    man = json.loads((DS / "manifest.json").read_text())
    clips = man["clips"]
    view = DS / "viewing"
    view.mkdir(exist_ok=True)
    font = ImageFont.load_default(size=13)
    for c in clips:  # DERIVATIVE: H.264 yuv420p for browser playback; never used for inference
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(DS / c["filename"]), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "14", "-movflags", "+faststart", str(view / f"{c['clip_id']}.mp4")], check=True)
    # contact sheet: one tile per clip = first frame with the true centre at t=0,1,..,5 s marked (derivative image)
    tile, lab = 192, 34
    sheet = Image.new("RGB", (4 * tile, 6 * (tile + lab)), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    for n, c in enumerate(clips):
        r, q = divmod(n, 4)
        fr = Image.fromarray(render_frame(c["x0"], c["y0"], c["g"], c["rgb"], 0))
        fd = ImageDraw.Draw(fr)
        for tt in range(1, 6):
            u, v = to_px(*center(c["x0"], c["y0"], c["g"], float(tt)))
            fd.ellipse((u - 4, v - 4, u + 4, v + 4), outline=(255, 255, 255), width=1)
            fd.text((u + 6, v - 6), f"{tt}s", fill=(255, 255, 255), font=font)
        ux, vy = to_px(c["x0"] + 8, c["y0"] - 8 * c["g"])
        fd.line([(ux - 7, vy - 7), (ux + 7, vy + 7)], fill=(255, 255, 0), width=2); fd.line([(ux - 7, vy + 7), (ux + 7, vy - 7)], fill=(255, 255, 0), width=2)
        x, y = q * tile, r * (tile + lab)
        sheet.paste(fr.resize((tile, tile), Image.NEAREST), (x, y + lab))
        d.text((x + 4, y + 2), f"{c['clip_id']}", fill=(255, 255, 255), font=font)
        d.text((x + 4, y + 18), f"scene {c['scene']} g={c['g']} {c['color']}", fill=(200, 200, 200), font=font)
    sheet.save(DS / "contact_sheet.png")
    rows = "\n".join(
        f"<tr><td>{c['clip_id']}</td><td>{c['scene']}</td><td>({c['x0']}, {c['y0']})</td><td>{c['g']}</td><td>{c['color']}</td>"
        f"<td><video src='viewing/{c['clip_id']}.mp4' width=192 height=192 controls loop muted preload=metadata></video></td>"
        f"<td><a href='viewing/{c['clip_id']}.mp4'>open (derivative mp4)</a><br><a href='{c['filename']}' download>download original FFV1 mkv</a><br>"
        f"<code>{c['sha256'][:16]}…</code></td></tr>" for c in clips)
    ms = sha256_file(DS / "manifest.json")
    (DS / "gallery.html").write_text(f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Gravity pilot - 24 clips</title>
<style>body{{font:14px system-ui;margin:16px;background:#fff;color:#111}}td,th{{border:1px solid #ccc;padding:6px;vertical-align:middle}}table{{border-collapse:collapse}}img{{max-width:100%}}
@media(prefers-color-scheme:dark){{body{{background:#161616;color:#eee}}td,th{{border-color:#444}}a{{color:#8ab4ff}}}}</style>
<h1>24-video gravity pilot - dataset for approval</h1>
<p><b>Protocol v1.0. Dataset not yet approved. No model has been run.</b> Manifest SHA-256: <code>{ms}</code></p>
<p>Each clip: 384x384, 30 fps, 151 frames (5 s simulated), one ball. Experimental clips are the lossless FFV1 <code>.mkv</code> files in <code>videos/</code>;
the <code>.mp4</code> files in <code>viewing/</code> are <b>derivative H.264 copies for browser viewing only and never enter inference</b>.</p>
<p>Clip counts: 6 start positions x 2 gravities (9.8, 4.9 m/s^2) x 2 colours. Colour pairs share identical trajectories. Encoder would see only the first 2 s (frames 0-60).</p>
<h2>Contact sheet</h2><p>First frame of each clip; white rings = true centre at 1-5 s; yellow X = target at 4 s (derivative image). Rows are scenes 0-5; columns: g=9.8 orange, g=9.8 blue, g=4.9 orange, g=4.9 blue.</p>
<p><a href="contact_sheet.png"><img src="contact_sheet.png" alt="contact sheet" width=768></a></p>
<h2>All 24 clips</h2>
<table><tr><th>clip</th><th>scene</th><th>start (x0,y0) m</th><th>g</th><th>colour</th><th>preview</th><th>links / sha256</th></tr>
{rows}</table>""")
    print("gallery written; manifest sha256", ms)


# ---------------------------------------------------------------- approval gate / stage B stubs
def receipt_ok():
    r = DS / "approval_receipt.json"
    if not r.exists():
        return False, "no approval_receipt.json"
    rec = json.loads(r.read_text())
    cur = sha256_file(DS / "manifest.json")
    if rec.get("manifest_sha256") != cur:
        return False, "receipt manifest hash does not match current manifest (dataset edited?)"
    return True, "ok"


def cmd_record_approval(a):
    cur = sha256_file(DS / "manifest.json")
    if a.manifest_sha256 != cur:
        sys.exit(f"refused: given hash != current manifest hash {cur}")
    (DS / "approval_receipt.json").write_text(json.dumps({"manifest_sha256": cur, "approver": a.approver, "note": a.note,
                                                          "approved_utc": a.utc}, indent=1) + "\n")
    print("receipt written")


def cmd_extract(_):
    ok, why = receipt_ok()
    if not ok:
        sys.exit(f"REFUSED: dataset not approved ({why}). Stage B is locked.")
    sys.exit("Stage B (extract) is not implemented in this commit.")


def cmd_stub(a):
    sys.exit(f"'{a.cmd}' is Stage B and is not implemented.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n, f in (("generate", cmd_generate), ("validate", cmd_validate), ("gallery", cmd_gallery), ("extract", cmd_extract),
                 ("analyze", cmd_stub), ("report", cmd_stub)):
        sp.add_parser(n).set_defaults(fn=f)
    r = sp.add_parser("record-approval")
    r.add_argument("--manifest-sha256", required=True); r.add_argument("--approver", required=True)
    r.add_argument("--utc", required=True, help="approval time, supplied by the human, e.g. 2026-10-10T12:00:00Z"); r.add_argument("--note", default="")
    r.set_defaults(fn=cmd_record_approval)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
