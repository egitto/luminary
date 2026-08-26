#!/usr/bin/env python3
"""Capture what a pattern looks like, from the live web client.

Drives the real UI with Playwright, so the pixels come through the production
path: engine -> codec -> WebSocket -> JS decoder -> canvas. Defaults to the
cloth-glow render and refuses to shoot if WebGL2 did not come up, because the
client's fallback to flat cells is silent and looks like a legitimate frame.

Three outputs, because two different questions get asked of a pattern:

    sheet   N frames across a span, tiled into ONE labelled image
    still   one frame at an exact t
    clip    a video

These serve two different readers, and neither substitutes for the other.

  * `sheet` is for whoever is *iterating* -- typically an agent in a cloud
    worker, which cannot watch a video at all. One file, one look, the whole
    arc of the pattern, and cheap: each tile is its own keyframe rather than a
    point in a delta chain, so sampling an hour costs what sampling a second
    does. This is the default reach while a pattern is being written.
  * `clip` is for whoever is *deciding* -- a human watching a pattern move
    before merging it. Motion is the thing being judged and a grid cannot show
    it. The send copy is kept small because it has to cross a metered playa
    link to reach them; the full-resolution master stays where it was made
    unless someone asks for it.

    python3 scripts/capture.py sheet --pattern life --lights pentagon-4A-37 \
        --at 60 --span 30 -o sheet.jpg
    python3 scripts/capture.py still --pattern aurora --at 150.75 -o shot.png
    python3 scripts/capture.py clip  --pattern life --at 55 --span 30 -o clip.webm

What each output is faithful to differs, and the difference matters:

  * `sheet` and `still` resync before every frame, so each is a keyframe: the
    pattern's ground truth at that t, with no delta history behind it.
  * `clip` steps the codec at the server's own frame rate, so the chain of
    deltas -- and every artifact the installation would show -- is the real
    one. The rate is read off the socket, never passed in.

`sheet` and `clip` print frame statistics, so a capture can be judged without
opening it (a `still` is one frame and has none to give). Video mode also writes a full-resolution master beside the small copy,
named `<stem>_full_size_send_only_if_asked.<ext>` -- the label is the point, on
a metered link the master should never be the thing casually attached.

Needs a server already running:  python -m luminary.cli serve --seed-demo
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import math
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, List, Sequence

import numpy as np
from playwright.sync_api import sync_playwright

# This container's browser install has a layout Playwright's own resolver
# misses, so this pin is a workaround for that and nothing more -- anywhere
# else, Playwright finds its own chromium and this path simply will not exist.
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
# Headless chromium has no GPU; ANGLE over SwiftShader gives it real WebGL2.
GL_ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


# --------------------------------------------------------------------- tooling


def find_ffmpeg() -> str:
    """Only `sheet` and `clip` call this; `still` needs no ffmpeg at all."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    raise SystemExit(
        "sheet and clip need ffmpeg on PATH (with libvpx-vp9 and drawtext). "
        "`still` works without it."
    )


def probe_size(ffmpeg: str, path: Path) -> tuple[int, int]:
    """Pixel size of one frame, straight from ffmpeg's own decode."""
    out = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
    ).stderr
    for token in out.replace(",", " ").split():
        if "x" in token and token.replace("x", "").isdigit():
            w, h = token.split("x")
            return int(w), int(h)
    raise SystemExit(f"could not read the frame size of {path}")


# ------------------------------------------------------------------ the client


def open_page(pw: Any, a: Any) -> tuple[Any, Any, tuple[str, str]]:
    """A browser on the live UI, played, with the glow render confirmed up."""
    launch: dict[str, Any] = {"args": GL_ARGS}
    if Path(CHROME).exists():
        launch["executable_path"] = CHROME
    try:
        browser = pw.chromium.launch(**launch)
    except Exception as exc:
        raise SystemExit(
            f"could not start chromium: {exc}\nTry `playwright install chromium`."
        ) from exc
    context = browser.new_context(
        viewport={"width": a.width, "height": a.height}, device_scale_factor=1
    )
    page = context.new_page()
    try:
        page.goto(f"http://127.0.0.1:{a.port}/", wait_until="networkidle")
    except Exception as exc:
        browser.close()
        raise SystemExit(
            f"no server on port {a.port}: {type(exc).__name__}. Start one with "
            "`python -m luminary.cli serve --seed-demo`."
        ) from exc
    page.wait_for_function(
        "() => document.getElementById('pattern').options.length > 0"
    )

    def pick(sel_id: str, want: str) -> str:
        """Resolve a name to one option, or refuse.

        Exactness first, and ambiguity is fatal. The old order -- first option
        whose value matched OR whose label merely contained the string -- let
        `--pattern wave` select `ripple` (its description reads "circular
        waves") and write it to wave.png. Capturing a different pattern than
        the one named is worse than any error message: everything downstream
        is of the wrong thing and the only tell is one word in a printed label.
        """
        opts = page.eval_on_selector_all(
            f"#{sel_id} option", "os => os.map(o => [o.value, o.textContent])"
        )
        exact = [(v, l) for v, l in opts if want == v]
        if not exact:
            exact = [
                (v, l)
                for v, l in opts
                if want.lower() == l.split(" \u2014 ")[0].lower()
            ]
        loose = exact or [(v, l) for v, l in opts if want.lower() in l.lower()]
        if not loose:
            names = ", ".join(sorted(v for v, _ in opts))
            raise SystemExit(f"no #{sel_id} option matching {want!r}. have: {names}")
        if len({l for _, l in loose}) > 1:
            hits = ", ".join(f"{v} ({l})" for v, l in loose)
            raise SystemExit(
                f"{want!r} matches {len(loose)} different #{sel_id} options: "
                f"{hits}. Name one exactly."
            )
        if len(loose) > 1:
            # Same label, different ids -- the store seeded the same geometry
            # twice. The user cannot tell them apart and neither can the
            # capture, so picking one is honest as long as it is said out loud.
            print(
                f"  ({len(loose)} entries named {loose[0][1]!r}; using {loose[0][0]})"
            )
        page.select_option(f"#{sel_id}", loose[0][0])
        return str(loose[0][1])

    labels = (pick("lights", a.lights), pick("pattern", a.pattern))
    # Drop the chrome after the selects are set (they live in the header) but
    # before Play: the client sizes its canvases once, in buildDrawList().
    page.add_style_tag(content="header{display:none!important}")
    page.evaluate("document.getElementById('play').click()")
    page.wait_for_function(
        "() => document.getElementById('status').textContent === 'connected'",
        timeout=30000,
    )
    page.wait_for_timeout(1500)
    if page.is_disabled("#render"):
        raise SystemExit(
            "WebGL2 never came up -- the client fell back to flat cells, "
            "so there is no glow render to capture."
        )
    # Stop the free-running repaint: it exists to keep glow.params tunable from
    # the console, but under software GL it re-renders the same frame until the
    # rAF period stretches to hundreds of ms and everything else queues behind.
    page.evaluate("window.client.captureMode = true")
    page.evaluate("window.client.send({type: 'pause'})")
    return browser, page, labels


def wire_fps(page: Any) -> float:
    """The rate the server built its Engine at, read off the live socket.

    Never a flag. A capture that steps the codec at some other rate carries
    different deltas, and where those hit the codec's per-frame correction
    limits (spec §11.4.3) they show as different artifacts. A flag
    documented as "must match the installation" is a trap the moment the two
    drift. The socket URL cannot be wrong.
    """
    url = str(page.evaluate("window.client.ws.url"))
    for part in url.partition("?")[2].split("&"):
        key, _, value = part.partition("=")
        if key == "fps":
            return float(value)
    raise SystemExit(f"no fps in the socket URL {url!r}")


JS_CAPTURE = """
async ({t, fmt, quality, keep, resync}) => {
  const c = window.client;
  // A resync makes the next frame a keyframe, so what comes back is the
  // pattern at t rather than a point in a delta chain. Wire-faithful capture
  // (a clip) must NOT do this; ground-truth capture (a still, a sheet) must.
  if (resync) c.send({type: 'resync'});
  const before = c.applied;
  c.send({type: 'step', t});
  // Wait for the stepped frame's bytes to be applied. `applied` is monotonic;
  // a paint counter cannot serve, because paints happen on their own clock.
  const deadline = performance.now() + 30000;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  while (c.applied <= before) {
    if (performance.now() > deadline) throw new Error('step ' + t + ' never applied');
    await sleep(2);
  }
  // A keyframe spans several messages; let the queue go quiet so a
  // half-applied board is never what gets painted.
  let seen = c.applied;
  for (;;) {
    await sleep(12);
    if (c.applied === seen) break;
    seen = c.applied;
  }
  if (!keep) return null;   // wire advanced; this frame is not being saved
  c.paint();
  // Same JS task as the WebGL render, so the drawing buffer is still valid
  // and no preserveDrawingBuffer is needed.
  const glow = document.getElementById('glow');
  const over = document.getElementById('canvas');
  const w = over.width, h = over.height;
  let buf = window.__captureBuf;
  if (!buf || buf.width !== w || buf.height !== h) {
    buf = window.__captureBuf = document.createElement('canvas');
    buf.width = w; buf.height = h;
  }
  const ctx = buf.getContext('2d');
  ctx.clearRect(0, 0, w, h);
  if (glow.width) ctx.drawImage(glow, 0, 0, w, h);
  ctx.drawImage(over, 0, 0);
  return buf.toDataURL(fmt === 'jpeg' ? 'image/jpeg' : 'image/png', quality / 100);
}
"""


def _write(out_dir: Path, index: int, ext: str, data_url: str) -> None:
    payload = data_url.split(",", 1)[1]
    (out_dir / f"f{index:05d}.{ext}").write_bytes(base64.b64decode(payload))


def grab(page: Any, times: Sequence[float], out_dir: Path, a: Any) -> int:
    """Keyframe-exact frames at arbitrary t -- the pattern's ground truth.

    Costs one step per frame however far apart the t values are, because
    patterns are pure functions of (lights, t) (spec §9.1.3) and a keyframe
    carries no history. Sampling an hour of a pattern is as cheap as sampling
    a second of it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = "jpg" if a.format == "jpeg" else "png"
    for i, t in enumerate(times):
        data_url = page.evaluate(
            JS_CAPTURE,
            {
                "t": t,
                "fmt": a.format,
                "quality": a.quality,
                "keep": True,
                "resync": True,
            },
        )
        _write(out_dir, i, ext, data_url)
    return len(times)


def step_frames(
    page: Any,
    out_dir: Path,
    start: float,
    fps: float,
    lo: int,
    hi: int,
    every: int,
    warmup: int,
    a: Any,
    note: Callable[[int], None] | None = None,
) -> int:
    """Drive the wire one frame at a time and read the pixels out.

    Wall clock is irrelevant: the page is asked for a specific t and hands back
    the frame that answers it. A renderer managing 2 fps still yields a 30 fps
    clip; it just takes longer to make.

    Everything for one frame happens inside a single page.evaluate: step, wait
    for the bytes, paint, composite, encode. Pixels leave through the canvas
    rather than a screenshot because screenshotting the glow costs ~2 s a frame
    against a ~300 ms render -- a fixed per-composite price in the software-GL
    path that does not shrink with resolution. Reading the canvas in the same
    JS task as the render skips it entirely.

    Frames between kept ones are still stepped, never skipped: dropping them
    would change what each delta has to carry, and with it the artifacts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = "jpg" if a.format == "jpeg" else "png"
    # Jumping to an arbitrary t is a discontinuity the dead-reckoning codec
    # must not try to predict through.
    page.evaluate("window.client.send({type: 'resync'})")
    # Warm up into the slice. A worker's first frame rides a fresh keyframe,
    # but the deltas after it settle over a few frames, and a slice boundary
    # should not be where that shows.
    for w in range(warmup * every, 0, -1):
        page.evaluate(
            JS_CAPTURE,
            {
                "t": start + (lo - w) / fps,
                "fmt": a.format,
                "quality": a.quality,
                "keep": False,
                "resync": False,
            },
        )
    kept = 0
    for i in range(lo, hi):
        keep = i % every == 0
        data_url = page.evaluate(
            JS_CAPTURE,
            {
                "t": start + i / fps,
                "fmt": a.format,
                "quality": a.quality,
                "keep": keep,
                "resync": False,
            },
        )
        if not keep:
            continue
        _write(out_dir, i // every, ext, data_url)
        kept += 1
        if note is not None:
            note(1)
    return kept


def clip_frames(a: Any, out_dir: Path) -> tuple[int, float, tuple[str, str]]:
    """Step the whole clip, split across `--workers` independent browsers.

    Slicing the timeline is only sound because patterns are pure functions of
    (lights, t) (spec §9.1.3) -- a worker can start anywhere without replaying
    what came before. Each browser gets its own socket, and the server builds a
    fresh Engine per connection, so the slices never share codec state. One
    software-GL context leaves most of the machine idle; four fill it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # One browser opens first, alone, to answer two questions the rest depend
    # on: what rate the wire runs at, and whether the glow came up at all.
    with sync_playwright() as pw:
        browser, page, labels = open_page(pw, a)
        rate = wire_fps(page)
        browser.close()

    every = max(1, round(rate / a.fps))
    # Keeping one frame in `every` and then encoding at --fps only tells the
    # truth when the two agree. At 20 fps off a 30 fps wire, every=2 keeps 15
    # frames a second and plays them at 20: a 4 s span becomes a 3 s clip
    # running 1.33x fast, silently. Refuse instead of lying about the speed.
    exact = rate / every
    if abs(exact - a.fps) > 1e-6:
        usable = ", ".join(
            f"{rate / k:g}"
            for k in range(1, int(rate) + 1)
            if float(rate / k).is_integer()
        )
        raise SystemExit(
            f"--fps {a.fps:g} does not divide the {rate:g} fps wire: the clip would "
            f"play {a.fps / exact:.2f}x speed. Usable rates here: {usable}"
        )
    total = int(round(a.span * rate)) // every
    if total < 1:
        raise SystemExit(f"--span {a.span} at {a.fps} fps is less than one frame")
    workers = a.workers or max(1, min(4, (os.cpu_count() or 2) // 2))
    # Slice on kept-frame boundaries so every worker starts on a saved frame.
    bounds = [round(k * total / workers) * every for k in range(workers + 1)]
    errors: List[BaseException] = []
    done = [0]
    progress = threading.Lock()

    def note(k: int) -> None:
        with progress:
            done[0] += k
            if done[0] % 100 < k:
                print(f"  ... {done[0]}/{total} frames", flush=True)

    def slice_worker(lo: int, hi: int) -> None:
        if lo >= hi:
            return
        try:
            with sync_playwright() as pw:
                browser, page, _ = open_page(pw, a)
                try:
                    step_frames(
                        page, out_dir, a.at, rate, lo, hi, every, a.warmup, a, note
                    )
                finally:
                    browser.close()
        except BaseException as exc:  # surface it after the join
            errors.append(exc)

    threads = [
        threading.Thread(target=slice_worker, args=(bounds[k], bounds[k + 1]))
        for k in range(workers)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:
        raise errors[0]
    return total, rate, labels


# ------------------------------------------------------------------- judgement


def frame_stats(ffmpeg: str, frames: Path, ext: str) -> str:
    """One line describing the frames, so a capture can be judged unopened.

    These are the two numbers every failure of this harness showed up in: a
    high duplicate fraction means the page handed back a frame it had not
    repainted, and motion with a large spread relative to its mean means
    uneven frame timing -- which is what "shaky" is, when you measure it.
    """
    files = sorted(frames.glob(f"f*.{ext}"))
    if len(files) < 2:
        return f"{len(files)} frame"
    digests = [hashlib.sha1(f.read_bytes()).digest() for f in files]
    dupes = sum(1 for a, b in zip(digests, digests[1:]) if a == b)
    raw = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(frames / f"f%05d.{ext}"),
            "-vf",
            "scale=128:128",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        capture_output=True,
    ).stdout
    gray = np.frombuffer(raw, np.uint8).reshape(-1, 128 * 128).astype(np.int16)
    motion = np.abs(np.diff(gray, axis=0)).mean(axis=1)
    return (
        f"{len(files)} frames · {dupes / (len(files) - 1):.1%} duplicated · "
        f"motion {motion.mean():.2f} ± {motion.std():.2f} /px"
    )


# --------------------------------------------------------------------- outputs


def contact_sheet(
    ffmpeg: str, frames: Path, ext: str, times: Sequence[float], dst: Path, a: Any
) -> None:
    """Tile every frame into one labelled image.

    The label is not decoration: without it the reader has a grid of pictures
    and no way to say which moment any of them is, which is most of what a
    sheet is for.
    """
    src_w, src_h = probe_size(ffmpeg, frames / f"f00000.{ext}")
    cw = a.cell - a.cell % 2
    ch = int(round(cw * src_h / src_w)) // 2 * 2
    cols = min(a.cols, len(times))
    rows = math.ceil(len(times) / cols)
    pad = 6
    if not Path(FONT).exists():
        print(f"  (no font at {FONT} — tiles go out unlabelled)")
    stamps = ",".join(
        f"drawtext=fontfile={FONT}:text='{t:.2f}s'"
        f":x={(i % cols) * (cw + pad) + 10}:y={(i // cols) * (ch + pad) + 8}"
        f":fontsize=20:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=5"
        for i, t in enumerate(times)
        if Path(FONT).exists()
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(frames / f"f%05d.{ext}"),
            "-vf",
            f"scale={cw}:{ch},tile={cols}x{rows}:padding={pad}:color=#0b0b0b"
            + (f",{stamps}" if stamps else ""),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(dst),
        ],
        check=True,
    )


def assemble(
    ffmpeg: str,
    frames: Path,
    ext: str,
    dst: Path,
    fps: int,
    width: int | None,
    crf: int,
) -> None:
    """Encode a directory of stepped frames at exactly the intended rate.

    No dedup, no re-timing: every frame here is a distinct t that was actually
    rendered, so the rate it goes out at is the rate it was made at.
    """
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frames / f"f%05d.{ext}"),
    ]
    if width:
        cmd += ["-vf", f"scale={width}:-2"]  # -2 keeps the aspect and stays even
    # CRF with -b:v 0 is VP9's constant-quality mode. Glow on black is mostly
    # flat dark field, which VP9 gives away nearly free.
    cmd += [
        "-c:v",
        "libvpx-vp9",
        "-crf",
        str(crf),
        "-b:v",
        "0",
        "-row-mt",
        "1",
        "-deadline",
        "good",
        "-cpu-used",
        "2",
        "-r",
        str(fps),
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def human(path: Path) -> str:
    n = path.stat().st_size
    return f"{n / 1e6:.2f} MB" if n >= 1e6 else f"{n / 1e3:.0f} kB"


# ------------------------------------------------------------------------ main


def main() -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pattern", default="aurora")
    common.add_argument("--lights", default="hex-demo", help="geometry name or id")
    common.add_argument("--port", type=int, default=8080)
    common.add_argument("--at", type=float, default=3.0, help="pattern seconds")
    # Measured on a 6 s clip: 1280x900 87 s, 960x675 74 s, 800x563 68 s. The
    # per-frame cost is mostly step-and-decode, not pixels, so shrinking the
    # viewport buys little -- and costs a little, since the send copy loses
    # the downscale averaging that was making it smaller.
    common.add_argument("--width", type=int, default=1280, help="browser viewport")
    common.add_argument("--height", type=int, default=900)
    common.add_argument(
        "--format",
        choices=["png", "jpeg"],
        default="png",
        help="how frames leave the page; jpeg encodes faster",
    )
    common.add_argument("--quality", type=int, default=92, help="jpeg quality")
    common.add_argument("-o", "--out", type=Path, required=True)

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)

    sub.add_parser("still", parents=[common], help="one frame at an exact t")

    sheet = sub.add_parser("sheet", parents=[common], help="N frames, one image")
    sheet.add_argument(
        "--span",
        type=float,
        default=30.0,
        help="pattern seconds the tiles are spread across",
    )
    sheet.add_argument("--tiles", type=int, default=12)
    sheet.add_argument("--cols", type=int, default=4)
    sheet.add_argument("--cell", type=int, default=420, help="tile width in px")

    clip = sub.add_parser("clip", parents=[common], help="a video")
    clip.add_argument("--span", type=float, default=10.0, help="clip length in seconds")
    clip.add_argument("--fps", type=int, default=15, help="output frame rate")
    clip.add_argument("--send-width", type=int, default=800)
    clip.add_argument("--send-crf", type=int, default=34)
    clip.add_argument("--master-crf", type=int, default=20)
    clip.add_argument(
        "--workers",
        type=int,
        default=0,
        help="browsers rendering disjoint slices; default cores/2, max 4",
    )
    clip.add_argument(
        "--warmup",
        type=int,
        default=4,
        help="frames a worker renders and throws away before its slice",
    )
    clip.add_argument("--keep-frames", action="store_true")

    a = ap.parse_args()
    ffmpeg = "" if a.mode == "still" else find_ffmpeg()
    ext = "jpg" if a.format == "jpeg" else "png"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    # Clear, not just create. Frames are read back as f%05d, so a shorter run
    # into the same output would splice the tail of a longer one onto its own
    # and exit 0 -- and the run that leaves a directory behind is the one that
    # crashed, which makes the retry the corrupted one.
    work = a.out.parent / f".capture-{a.out.stem}"
    shutil.rmtree(work, ignore_errors=True)

    if a.mode == "clip":
        n, rate, labels = clip_frames(a, work)
        print(f"{labels[1]} on {labels[0]} · wire {rate:g} fps · {n} frames")
        print(f"  {frame_stats(ffmpeg, work, ext)}")
        master = a.out.with_name(
            f"{a.out.stem}_full_size_send_only_if_asked{a.out.suffix}"
        )
        assemble(ffmpeg, work, ext, master, a.fps, None, a.master_crf)
        assemble(ffmpeg, work, ext, a.out, a.fps, a.send_width, a.send_crf)
        print(f"  send   {a.out}  ({human(a.out)})")
        print(f"  master {master}  ({human(master)})")
        if not a.keep_frames:
            shutil.rmtree(work, ignore_errors=True)
        return 0

    if a.mode == "sheet":
        if a.tiles < 1:
            raise SystemExit("--tiles must be at least 1")
        step = a.span / (a.tiles - 1) if a.tiles > 1 else 0.0
        times = [a.at + i * step for i in range(a.tiles)]
    else:
        times = [a.at]

    with sync_playwright() as pw:
        browser, page, labels = open_page(pw, a)
        try:
            grab(page, times, work, a)
        finally:
            browser.close()

    print(f"{labels[1]} on {labels[0]} · keyframe-exact")
    if a.mode == "sheet":
        print(f"  {frame_stats(ffmpeg, work, ext)}")
        print(f"  t = {times[0]:.2f}s .. {times[-1]:.2f}s, every {step:.2f}s")
        contact_sheet(ffmpeg, work, ext, times, a.out, a)
    else:
        shutil.copyfile(work / f"f00000.{ext}", a.out)
    shutil.rmtree(work, ignore_errors=True)
    print(f"  {a.out}  ({human(a.out)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
