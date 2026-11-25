#!/usr/bin/env python3
"""One scan pass for the web UI, in its own process.

The server starts this instead of calling the core in-process so a cancellation
is a plain ``terminate()`` and an O(n^2) comparison cannot stall HTTP requests.

Protocol: stdout carries one JSON object per line. A large pair graph is written
to the ``--result`` file instead of a single stdout line.

    {"event": "progress", "stage": "hashing", "done": 3, "total": 120, "current": "a.jpg"}
    {"event": "result", "path": "...", "pairs": 12}
    {"event": "error", "error": "..."}

Everything the core prints (verbose warnings, tqdm, "No image files found") is
redirected to stderr so it can never corrupt the protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import similar_image_remover as core  # noqa: E402  (path set up above)


def _relative(path: str, dataset: str) -> str:
    """Dataset-relative POSIX key, the only path form the server accepts."""
    return Path(os.path.relpath(path, dataset)).as_posix()


def scan(
    directory: str,
    config: Dict,
    progress: Optional[Callable[[str, int, int, str], None]] = None,
) -> Dict:
    """Run the hash/compare/ORB pipeline and return a serialisable result.

    Pairs are cached down to ``pair_floor`` (0.5 by default) so the UI can move
    the similarity threshold without rehashing. When ORB is enabled it therefore
    verifies every pair above the floor, which is the explicit cost of instant
    threshold changes; the UI documents that.
    """
    dataset = os.path.abspath(directory)
    methods = list(config.get("hash_methods") or ["ahash", "phash"])
    pair_floor = float(config.get("pair_floor", 0.5))
    requested = float(config.get("threshold", 0.90))
    scan_threshold = max(0.0, min(1.0, min(requested, pair_floor)))

    similar = core.find_similar_images(
        dataset,
        threshold=scan_threshold,
        hash_methods=methods,
        hash_size=int(config.get("hash_size", 32)),
        use_gpu=bool(config.get("use_gpu", False)),
        use_orb_verification=bool(config.get("verify_orb", False)),
        orb_min_matches=int(config.get("orb_min_matches", 10)),
        orb_threshold=float(config.get("orb_threshold", 0.8)),
        verbose=bool(config.get("verbose", False)),
        progress=progress,
    )

    pairs = []
    seen = set()
    for path_a, matches in similar.items():
        for path_b, similarity in matches:
            key = tuple(sorted((path_a, path_b)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                [
                    _relative(path_a, dataset),
                    _relative(path_b, dataset),
                    round(float(similarity), 6),
                ]
            )
    pairs.sort()

    return {
        "dataset": dataset,
        "pairs": pairs,
        "scanned_at": time.time(),
        "config": {
            **config,
            "hash_methods": methods,
            "pair_floor": pair_floor,
            "scan_threshold": scan_threshold,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="scan config JSON file")
    parser.add_argument("--result", required=True, help="where to write the result JSON")
    args = parser.parse_args(argv)

    try:
        config = json.loads(Path(args.config).read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"bad config: {exc}", file=sys.stderr)
        return 2

    # `protocol` is the real stdout. Anything else that writes to stdout (the
    # core's prints, tqdm is disabled by passing a progress callback) goes to
    # stderr, where the server collects a diagnostic tail.
    protocol = sys.stdout
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    def emit(obj: Dict) -> None:
        protocol.write(json.dumps(obj) + "\n")
        protocol.flush()

    def progress(stage: str, done: int, total: int, current: str) -> None:
        emit(
            {
                "event": "progress",
                "stage": stage,
                "done": int(done),
                "total": int(total),
                "current": os.path.basename(current),
            }
        )

    try:
        result = scan(config["directory"], config.get("config", {}), progress)
        result_path = Path(args.result)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = result_path.with_suffix(result_path.suffix + ".tmp")
        tmp.write_text(json.dumps(result))
        tmp.replace(result_path)
        emit(
            {
                "event": "result",
                "path": str(result_path),
                "pairs": len(result["pairs"]),
            }
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        emit({"event": "error", "error": f"{type(exc).__name__}: {exc}"})
        return 1
    finally:
        sys.stdout = real_stdout


if __name__ == "__main__":
    sys.exit(main())
