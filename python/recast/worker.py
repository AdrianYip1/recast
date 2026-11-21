"""Long-running helper for the engine: keeps torch and the models loaded between steps.

python -m recast.worker QUEUE_DIR [--parent PID]

The engine drops `<id>.json` files ({"module": "recast" | "recast.object", "args": [...]}) into QUEUE_DIR.
Each one runs exactly like the command line would, in this process; its output goes to `<id>.log` and its
exit code to `<id>.done`. A `stop` file, or the parent process exiting, ends the worker.
Models load in the background as soon as it starts, so even the first step skips most of the wait.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

_busy = threading.Lock()   # one step at a time; the warm-up holds it while loading


def _parent_alive(pid: int | None) -> bool:
    if not pid:
        return True
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x00100000, False, pid)   # SYNCHRONIZE
        if not handle:
            return False
        try:
            return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == 0x102   # WAIT_TIMEOUT: still running
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _lower_priority() -> None:
    """Below the engine, so loading models never starves its light detection or frame rate."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00004000)   # BELOW_NORMAL_PRIORITY_CLASS
    else:
        os.nice(5)


def _warm_up() -> None:
    """Import and load everything the steps use, while the user is still picking photos."""
    with _busy:
        try:
            from .object.modes import generated
            from .object import segment
            import numpy as np
            generated.load_model()
            segment.salient_alpha(np.zeros((64, 64, 3), np.float32))
            from . import detect
            detect.run_models(np.zeros((64, 64, 3), np.uint8))
            import skimage.measure, trimesh                  # shape carving: a second's import on the first scene
            trimesh.creation.box().simplify_quadric_decimation(face_count=6)
            print("worker: models loaded", flush=True)
        except Exception as e:                       # a missing extra only matters when that step runs
            print(f"worker: warm-up skipped ({e})", flush=True)


def run_request(request: dict, log_path: Path) -> int:
    from . import cli
    from .object import cli as object_cli

    main = object_cli.main if request.get("module") == "recast.object" else cli.main
    with open(log_path, "w", encoding="utf-8", buffering=1) as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            main(list(request.get("args", [])))
            return 0
        except SystemExit as e:
            if e.code not in (None, 0):
                print(e.code if isinstance(e.code, str) else f"exit code {e.code}")
            return e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        except Exception:
            traceback.print_exc()
            return 1


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m recast.worker")
    p.add_argument("queue")
    p.add_argument("--parent", type=int, help="exit when this process does")
    args = p.parse_args(argv)
    queue = Path(args.queue)
    queue.mkdir(parents=True, exist_ok=True)
    _lower_priority()
    print(f"worker: ready, watching {queue}", flush=True)
    threading.Thread(target=_warm_up, daemon=True).start()

    while not (queue / "stop").exists() and _parent_alive(args.parent):
        requests = sorted(queue.glob("*.json"))
        if not requests:
            time.sleep(0.1)
            continue
        path = requests[0]
        running = path.with_suffix(".running")
        try:
            path.rename(running)
            request = json.loads(running.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            time.sleep(0.05)                          # still being written
            continue
        with _busy:
            code = run_request(request, path.with_suffix(".log"))
        path.with_suffix(".done").write_text(str(code))
        running.unlink(missing_ok=True)
    import shutil
    shutil.rmtree(queue, ignore_errors=True)
    print("worker: stopped", flush=True)


if __name__ == "__main__":
    main()
