"""Screenshot a pattern playing in the browser demo, headlessly.

Boots the real stack end to end — `serve --seed-demo`, the canvas client,
the wire protocol over WebSocket, the JS decoder — drives it with headless
Chromium, and saves a PNG of the page after the pattern has streamed for a
few seconds. This is the "does it actually look right in the client?" check
for containers with no display; for a codec-free still frame, use the SVG
snippet in `patterns/README.md` instead.

Usage:
    python scripts/screenshot_pattern.py --pattern aurora -o aurora.png
    python scripts/screenshot_pattern.py --pattern ripple --lights <id> \
        --settle 5 --server http://localhost:8080

With no --server, a throwaway `python -m luminary.cli serve --seed-demo`
is started on --port with an isolated store and torn down afterwards.

Requires `pip install playwright` and a Chromium install. Playwright's own
managed browser is used when present; otherwise set $LUMINARY_CHROMIUM or
rely on the /opt/pw-browsers/chromium fallback (the Claude cloud
container's pre-installed build).

Exits non-zero (with the reason on stderr) if the client never connects,
no frames decode, or the page logs a console error.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

FALLBACK_CHROMIUM = "/opt/pw-browsers/chromium"


def wait_for_server(url: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except OSError:
            time.sleep(0.3)
    raise RuntimeError(f"server at {url} did not come up within {timeout}s")


def launch_chromium(pw):  # type: ignore[no-untyped-def]
    args = ["--no-sandbox"]
    try:
        return pw.chromium.launch(headless=True, args=args)
    except PlaywrightError:
        exe = os.environ.get("LUMINARY_CHROMIUM", FALLBACK_CHROMIUM)
        if not Path(exe).exists():
            raise
        return pw.chromium.launch(headless=True, args=args, executable_path=exe)


def screenshot(
    url: str, pattern: str | None, lights: str | None, out: Path, settle: float
) -> None:
    with sync_playwright() as pw:
        browser = launch_chromium(pw)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        errors: list[str] = []
        page.on(
            "console",
            lambda msg: errors.append(msg.text) if msg.type == "error" else None,
        )
        page.on("pageerror", lambda err: errors.append(str(err)))

        page.goto(url)
        page.wait_for_selector("#pattern option", state="attached", timeout=15_000)
        if lights:
            page.select_option("#lights", lights)
        if pattern:
            page.select_option("#pattern", pattern)
        chosen = page.eval_on_selector("#pattern", "el => el.value")

        page.click("#play")
        page.wait_for_function(
            "document.getElementById('status').textContent === 'connected'",
            timeout=15_000,
        )
        # #stats stays empty until decoded frames arrive (updated every 500 ms).
        page.wait_for_function(
            "/\\d+ fps/.test(document.getElementById('stats').textContent)",
            timeout=15_000,
        )
        time.sleep(settle)

        stats = page.eval_on_selector("#stats", "el => el.textContent")
        page.screenshot(path=str(out))
        browser.close()

    print(f"pattern:    {chosen}")
    print(f"stats:      {stats}")
    print(f"screenshot: {out}")
    if errors:
        for e in errors:
            print(f"console error: {e}", file=sys.stderr)
        raise RuntimeError("page logged console errors")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pattern", help="pattern name (default: first in the registry)")
    ap.add_argument("--lights", help="lights geometry id (default: first listed)")
    ap.add_argument("-o", "--output", default="pattern.png", type=Path)
    ap.add_argument(
        "--settle", type=float, default=3.0, help="seconds of playback before the shot"
    )
    ap.add_argument("--server", help="reuse a running server instead of starting one")
    ap.add_argument(
        "--port", type=int, default=8642, help="port for the throwaway server"
    )
    args = ap.parse_args()

    server: subprocess.Popen[bytes] | None = None
    tmp: tempfile.TemporaryDirectory[str] | None = None
    url = args.server
    try:
        if url is None:
            tmp = tempfile.TemporaryDirectory(prefix="luminary-shot-")
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "luminary.cli",
                    "--store",
                    tmp.name,
                    "serve",
                    "--port",
                    str(args.port),
                    "--seed-demo",
                ],
                cwd=Path(__file__).resolve().parent.parent,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            url = f"http://127.0.0.1:{args.port}"
            wait_for_server(f"{url}/api/patterns")
        screenshot(url, args.pattern, args.lights, args.output, args.settle)
        return 0
    except (RuntimeError, PlaywrightError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if server is not None:
            server.terminate()
            server.wait(timeout=10)
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    sys.exit(main())
