#!/usr/bin/env python3
"""Review server for the similar-image-remover web UI. Standard library only.

    python3 similar_image_remover.py --web --username me --password secret
    python3 web/server.py --username me --password secret

The dataset is optional at startup: with no ``--directory`` the page opens a
folder field, and the current folder can be changed at any time from the page.
Authentication is HTTP Basic Auth and is required unless ``--no-auth`` is passed
explicitly for this run.

State, thumbnails and previews live under ``runs/<dataset-slug>/``. Deletion is
permanent (``os.remove``), exactly like the CLI; the safety comes from the UI
(arm-twice on a tile, a confirmation for bulk actions) and from server-side
invariants that refuse to empty a similarity group or touch a pinned file.

Endpoints
    GET  /api/ping                 -> liveness + dataset + auth flag
    GET  /api/state                -> dataset, job, config, result availability
    POST /api/dataset              {"path": "..."} -> open/scan a folder
    POST /api/scan                 {...config} -> start a scan
    POST /api/cancel               -> stop the running scan
    GET  /api/results?threshold=.. -> groups at this threshold
    POST /api/protect              {"path": ..., "protected": bool}
    POST /api/delete               {"paths": [...], "threshold": ...}
    GET  /thumb/<rel>              -> cached 400 px JPEG
    GET  /preview/<rel>            -> cached 1280 px JPEG
    GET  /image/<rel>              -> the source image

Paths in requests are dataset-relative ("sub/x.jpg"), never absolute, so a
crafted request cannot read or delete outside the current dataset.
"""

from __future__ import annotations

import argparse
import base64
import errno
import hashlib
import hmac
import json
import mimetypes
import os
import random
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import similar_image_remover as core  # noqa: E402  (path set up above)

RUNS_ROOT = ROOT / "runs"
WEB_ROOT = Path(__file__).resolve().parent
INDEX_HTML = WEB_ROOT / "index.html"
WORKER = WEB_ROOT / "scan_worker.py"

# Pairs are cached down to the scan's own threshold, capped at this default
# floor. A scan at 0.9 stores pairs to 0.5 so the slider can move freely; a scan
# at 0.3 stores down to 0.3 so the user can actually review that threshold.
PAIR_FLOOR = 0.5

HASH_METHODS = [
    "ahash",
    "phash",
    "dhash",
    "dhash_vertical",
    "whash",
    "colorhash",
    "crop_resistant",
]

DEFAULT_CONFIG = {
    "threshold": 0.90,
    "hash_methods": ["ahash", "phash"],
    "hash_size": 32,
    "verify_orb": False,
    "orb_min_matches": 10,
    "orb_threshold": 0.8,
    "use_gpu": False,
    "verbose": False,
    "seed": None,
}

CFG: Dict = {}
CFG_LOCK = threading.RLock()
JOB_LOCK = threading.Lock()
STATE_LOCK = threading.RLock()
DERIVED_LOCKS: Dict[str, threading.Lock] = {}
DERIVED_LOCKS_LOCK = threading.Lock()
DERIVED_SEM = threading.BoundedSemaphore(4)

JOB = {
    "phase": "idle",  # idle | scanning | loading | done | error | cancelled
    "stage": "",
    "done": 0,
    "total": 0,
    "current": "",
    "error": None,
    "started": None,
    "finished": None,
}

STATES: Dict[str, Dict] = {}  # slug -> dataset state, also on disk
GENERATION = 0


# ---------------------------------------------------------------------------
# small helpers


def slug_for(dataset: Path) -> str:
    """Stable, human-readable cache directory name for an absolute dataset."""
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", dataset.name).strip("-") or "dataset"
    digest = hashlib.sha1(str(dataset.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{base}-{digest}"


def run_dir(dataset: Path) -> Path:
    return RUNS_ROOT / slug_for(dataset)


def state_path(dataset: Path) -> Path:
    return run_dir(dataset) / "state.json"


def now() -> float:
    return time.time()


def resolve_rel(base: Path, rel: str) -> Path:
    """Join a client-supplied relative path onto ``base``, refusing escapes."""
    rel = urllib.parse.unquote(str(rel)).strip().lstrip("/")
    if not rel or "\x00" in rel:
        raise ValueError("empty or invalid path")
    base_resolved = base.resolve()
    path = (base_resolved / rel).resolve()
    if path != base_resolved and base_resolved not in path.parents:
        raise ValueError("path escapes the dataset")
    return path


def rel_key(base: Path, path: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def quote_path(rel: str) -> str:
    return "/".join(urllib.parse.quote(segment) for segment in rel.split("/"))


def job_snapshot() -> Dict:
    with JOB_LOCK:
        return dict(JOB)


def set_job(**fields) -> None:
    with JOB_LOCK:
        JOB.update(fields)


def bump_generation() -> int:
    global GENERATION
    with JOB_LOCK:
        GENERATION += 1
        return GENERATION


def generation_current(generation: int, dataset: Path) -> bool:
    with JOB_LOCK:
        return (
            generation == GENERATION
            and CFG.get("dataset") is not None
            and Path(CFG["dataset"]).resolve() == dataset.resolve()
        )


# ---------------------------------------------------------------------------
# config validation
#
# The UI may only change the scan algorithm, never server-level options such as
# the bind address, credentials, or the state directory.


def normalise_config(raw: Optional[Dict], base: Dict) -> Tuple[Dict, Optional[str]]:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(base or {})
    raw = raw or {}

    for key in (
        "threshold",
        "hash_methods",
        "hash_size",
        "verify_orb",
        "orb_min_matches",
        "orb_threshold",
        "use_gpu",
        "verbose",
        "seed",
    ):
        if key in raw:
            cfg[key] = raw[key]

    try:
        cfg["threshold"] = float(cfg["threshold"])
    except (TypeError, ValueError):
        return cfg, "threshold must be a number"
    if not 0.0 <= cfg["threshold"] <= 1.0:
        return cfg, "threshold must be between 0.0 and 1.0"

    methods = cfg.get("hash_methods")
    if isinstance(methods, str):
        methods = [methods]
    if not isinstance(methods, list) or not methods:
        return cfg, "choose at least one hash method"
    methods = list(dict.fromkeys(str(m).strip() for m in methods))
    unknown = [m for m in methods if m not in HASH_METHODS]
    if unknown:
        return cfg, f"unknown hash method(s): {', '.join(unknown)}"
    cfg["hash_methods"] = methods

    try:
        cfg["hash_size"] = int(cfg["hash_size"])
    except (TypeError, ValueError):
        return cfg, "hash size must be an integer"
    if not 4 <= cfg["hash_size"] <= 128:
        return cfg, "hash size must be between 4 and 128"

    cfg["verify_orb"] = bool(cfg.get("verify_orb"))
    try:
        cfg["orb_min_matches"] = int(cfg.get("orb_min_matches", 10))
    except (TypeError, ValueError):
        return cfg, "ORB minimum matches must be an integer"
    if not 1 <= cfg["orb_min_matches"] <= 1000:
        return cfg, "ORB minimum matches must be between 1 and 1000"

    try:
        cfg["orb_threshold"] = float(cfg.get("orb_threshold", 0.8))
    except (TypeError, ValueError):
        return cfg, "ORB threshold must be a number"
    if not 0.0 <= cfg["orb_threshold"] <= 1.0:
        return cfg, "ORB threshold must be between 0.0 and 1.0"

    cfg["use_gpu"] = bool(cfg.get("use_gpu"))
    cfg["verbose"] = bool(cfg.get("verbose"))

    seed = cfg.get("seed")
    if seed in (None, ""):
        cfg["seed"] = None
    else:
        try:
            cfg["seed"] = int(seed)
        except (TypeError, ValueError):
            return cfg, "seed must be an integer or blank"

    cfg["pair_floor"] = min(PAIR_FLOOR, cfg["threshold"])
    return cfg, None


# ---------------------------------------------------------------------------
# dataset state


def _clean_rel(rel: str) -> Optional[str]:
    """Normalise a dataset-relative key, rejecting anything that can escape."""
    rel = str(rel).replace("\\", "/").strip().lstrip("/")
    if not rel or "\x00" in rel:
        return None
    parts = [part for part in rel.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        return None
    return "/".join(parts)


def _state_from_payload(
    dataset: Path,
    payload: Dict,
    config: Dict,
    seed_used: int,
    protected: Optional[List[str]] = None,
) -> Dict:
    pairs: List[List] = []
    seen = set()
    for row in payload.get("pairs", []):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            continue
        rel_a = _clean_rel(row[0])
        rel_b = _clean_rel(row[1])
        if rel_a is None or rel_b is None:
            continue
        try:
            similarity = float(row[2])
        except (TypeError, ValueError):
            continue
        key = tuple(sorted((rel_a, rel_b)))
        if key in seen:
            continue
        seen.add(key)
        pairs.append([rel_a, rel_b, similarity])
    pairs.sort(key=lambda row: (row[0], row[1]))

    rels = sorted({row[0] for row in pairs} | {row[1] for row in pairs})
    resolutions: Dict[str, List[int]] = {}
    sizes: Dict[str, int] = {}
    for rel in rels:
        absolute = dataset / rel
        width, height = core.get_image_resolution(str(absolute))
        resolutions[rel] = [width, height]
        try:
            sizes[rel] = absolute.stat().st_size
        except OSError:
            sizes[rel] = 0

    keep_pins = sorted(set(protected or []) & set(rels))
    cfg = dict(config)
    cfg["seed_used"] = int(seed_used)
    cfg["pair_floor"] = float(
        cfg.get("pair_floor", cfg.get("scan_threshold", PAIR_FLOOR))
    )
    return {
        "dataset": str(dataset.resolve()),
        "pairs": pairs,
        "scanned_at": float(payload.get("scanned_at", now())),
        "config": cfg,
        "protected": keep_pins,
        "resolutions": resolutions,
        "sizes": sizes,
    }


def prune_missing(dataset: Path, state: Dict) -> List[str]:
    """Drop paths that no longer exist and return them.

    The state is derived data: if the user deletes a file outside the tool, a
    group containing it would otherwise keep a dead tile and, worse, count the
    missing file when enforcing "never empty a group". Pruning on load and after
    deletions keeps both honest.
    """
    missing = []
    rels = {row[0] for row in state.get("pairs", [])}
    rels |= {row[1] for row in state.get("pairs", [])}
    rels |= set(state.get("protected", []))
    for rel in sorted(rels):
        if not (dataset / rel).is_file():
            missing.append(rel)
    if not missing:
        return []
    gone = set(missing)
    state["pairs"] = [
        row for row in state.get("pairs", []) if row[0] not in gone and row[1] not in gone
    ]
    state["protected"] = [p for p in state.get("protected", []) if p not in gone]
    for rel in gone:
        state.get("resolutions", {}).pop(rel, None)
        state.get("sizes", {}).pop(rel, None)
    return missing


def save_state(dataset: Path, state: Dict) -> None:
    directory = run_dir(dataset)
    directory.mkdir(parents=True, exist_ok=True)
    path = state_path(dataset)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)
    with STATE_LOCK:
        STATES[slug_for(dataset)] = state


def load_state(dataset: Path) -> Optional[Dict]:
    path = state_path(dataset)
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None
    if state.get("dataset") != str(dataset.resolve()):
        return None
    state.setdefault("pairs", [])
    state.setdefault("protected", [])
    state.setdefault("resolutions", {})
    state.setdefault("sizes", {})
    state.setdefault("config", dict(DEFAULT_CONFIG))
    if prune_missing(dataset, state):
        save_state(dataset, state)
    return state


def get_state(dataset: Path) -> Optional[Dict]:
    slug = slug_for(dataset)
    with STATE_LOCK:
        if slug in STATES:
            return STATES[slug]
    state = load_state(dataset)
    if state is not None:
        with STATE_LOCK:
            STATES[slug] = state
    return state


# ---------------------------------------------------------------------------
# results and deletion policy


def graph_at(dataset: Path, pairs: List[List], threshold: float) -> Dict[str, List[Tuple[str, float]]]:
    """The core's similarity dict shape, with absolute paths, at a threshold."""
    graph: Dict[str, List[Tuple[str, float]]] = {}
    for rel_a, rel_b, similarity in pairs:
        if similarity < threshold:
            continue
        abs_a = str(dataset / rel_a)
        abs_b = str(dataset / rel_b)
        graph.setdefault(abs_a, []).append((abs_b, similarity))
        graph.setdefault(abs_b, []).append((abs_a, similarity))
    return graph


def build_results(dataset: Path, state: Dict, threshold: float) -> Dict:
    """Groups, policy survivors, similarity and reclaimable bytes at a threshold."""
    requested = float(threshold)
    floor = float(state.get("config", {}).get("pair_floor", PAIR_FLOOR))
    effective = max(floor, min(1.0, requested))
    graph = graph_at(dataset, state.get("pairs", []), effective)

    if not graph:
        return {
            "groups": [],
            "threshold": effective,
            "requested_threshold": requested,
            "clamped": requested < floor,
            "pair_floor": floor,
            "protected": sorted(state.get("protected", [])),
            "scanned_at": state.get("scanned_at"),
            "config": state.get("config", {}),
            "stats": {"groups": 0, "files": 0, "reclaimable": 0},
        }

    seed_used = state.get("config", {}).get("seed_used")
    if seed_used is None:
        # Old state files may predate seed_used. Derive a stable fallback from
        # the dataset path so repeated requests cannot pick a different winner.
        seed_used = int(hashlib.sha1(str(dataset).encode()).hexdigest()[:8], 16)
        state.setdefault("config", {})["seed_used"] = seed_used
    keep, _ = core.select_files_to_keep(
        graph, seed=seed_used, report=lambda _message: None
    )
    keep_set = set(keep)
    groups = core._similarity_groups(graph)
    protected = set(state.get("protected", []))
    resolutions = state.get("resolutions", {})
    sizes = state.get("sizes", {})

    def rel_of(absolute: str) -> str:
        return Path(absolute).relative_to(dataset).as_posix()

    out = []
    for members in groups:
        rels = sorted(rel_of(member) for member in members)
        winner = next(
            (rel for rel in rels if str(dataset / rel) in keep_set), rels[0]
        )
        similarities = {}
        for member in members:
            rel = rel_of(member)
            similarities[rel] = max(
                (sim for _other, sim in graph.get(member, [])), default=0.0
            )

        areas = {}
        for rel in rels:
            width, height = resolutions.get(rel, [0, 0])
            areas[rel] = int(width) * int(height)
        best_area = max(areas.values())
        tied = sum(1 for area in areas.values() if area == best_area) > 1

        deletable = [rel for rel in rels if rel != winner and rel not in protected]
        reclaimable = sum(int(sizes.get(rel, 0)) for rel in deletable)
        out.append(
            {
                "paths": rels,
                "keep": winner,
                "similarity": similarities,
                "resolutions": {rel: resolutions.get(rel, [0, 0]) for rel in rels},
                "sizes": {rel: int(sizes.get(rel, 0)) for rel in rels},
                "protected": [rel for rel in rels if rel in protected],
                "deletable": deletable,
                "reclaimable": reclaimable,
                "tie": tied,
                "max_similarity": max(similarities.values(), default=0.0),
            }
        )

    out.sort(key=lambda group: (-group["reclaimable"], group["paths"][0]))
    return {
        "groups": out,
        "threshold": effective,
        "requested_threshold": requested,
        "clamped": requested < floor,
        "pair_floor": floor,
        "protected": sorted(protected),
        "scanned_at": state.get("scanned_at"),
        "config": state.get("config", {}),
        "stats": {
            "groups": len(out),
            "files": len({rel for group in out for rel in group["paths"]}),
            "reclaimable": sum(group["reclaimable"] for group in out),
        },
    }


def validate_deletions(
    dataset: Path, state: Dict, paths: List[str], threshold: float
) -> Tuple[List[str], Dict[str, str]]:
    """Check a delete request against the current result graph.

    The browser is not trusted: a stale tab or a hand-written request must not be
    able to delete every copy in a group or a pinned file. Returns the normalised
    paths and a mapping of path -> reason when the whole request must be refused.
    """
    if not isinstance(paths, list) or not paths:
        return [], {"_": "no paths supplied"}
    if len(paths) > 10000:
        return [], {"_": "too many paths in one request"}

    effective = max(
        float(state.get("config", {}).get("pair_floor", PAIR_FLOOR)),
        min(1.0, float(threshold)),
    )
    graph = graph_at(dataset, state.get("pairs", []), effective)
    groups = core._similarity_groups(graph)
    group_of: Dict[str, int] = {}
    members_by_group: Dict[int, List[str]] = {}
    for index, members in enumerate(groups):
        rels = []
        for member in members:
            rel = Path(member).relative_to(dataset).as_posix()
            group_of[rel] = index
            rels.append(rel)
        members_by_group[index] = rels

    protected = set(state.get("protected", []))
    normalised: List[str] = []
    errors: Dict[str, str] = {}
    for raw in paths:
        if not isinstance(raw, str):
            errors[str(raw)] = "not a path"
            continue
        try:
            resolved = resolve_rel(dataset, raw)
        except ValueError as exc:
            errors[raw] = str(exc)
            continue
        rel = rel_key(dataset, resolved)
        if rel not in group_of:
            errors[rel] = "not part of the current results; rescan and retry"
            continue
        if rel in protected:
            errors[rel] = "pinned; unpin it before deleting"
            continue
        normalised.append(rel)

    if errors:
        return [], errors

    wanted_by_group: Dict[int, set] = {}
    for rel in normalised:
        wanted_by_group.setdefault(group_of[rel], set()).add(rel)

    for index, wanted in wanted_by_group.items():
        # A file that is already missing cannot be the survivor. Count only
        # paths that exist right now.
        remaining = [
            rel
            for rel in members_by_group[index]
            if rel not in wanted and (dataset / rel).is_file()
        ]
        if not remaining:
            label = members_by_group[index][0]
            errors[label] = "refusing to delete every file in a similarity group"
    if errors:
        return [], errors

    return normalised, {}


def remove_derived(dataset: Path, rel: str) -> None:
    safe = rel.replace("/", "__")
    directory = run_dir(dataset)
    for kind in ("thumbs", "previews"):
        path = directory / kind / f"{safe}.jpg"
        try:
            path.unlink()
        except OSError:
            pass


def perform_deletions(dataset: Path, state: Dict, paths: List[str]) -> Dict:
    """Permanently remove the validated paths and keep derived data in sync."""
    deleted: List[str] = []
    missing: List[str] = []
    failed: Dict[str, str] = {}
    sizes = state.get("sizes", {})
    log_lines = []

    for rel in paths:
        target = resolve_rel(dataset, rel)
        try:
            size = sizes.get(rel, target.stat().st_size)
            target.unlink()
            deleted.append(rel)
            log_lines.append(
                {"path": rel, "at": now(), "bytes": int(size)}
            )
        except FileNotFoundError:
            missing.append(rel)
        except OSError as exc:
            failed[rel] = str(exc)

    removed = set(deleted) | set(missing)
    if removed:
        state["pairs"] = [
            row
            for row in state.get("pairs", [])
            if row[0] not in removed and row[1] not in removed
        ]
        state["protected"] = [
            rel for rel in state.get("protected", []) if rel not in removed
        ]
        for rel in removed:
            state.get("resolutions", {}).pop(rel, None)
            state.get("sizes", {}).pop(rel, None)
            remove_derived(dataset, rel)

        if log_lines:
            log_path = run_dir(dataset) / "deleted.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                for entry in log_lines:
                    handle.write(json.dumps(entry) + "\n")

        prune_missing(dataset, state)
        save_state(dataset, state)

    return {
        "deleted": deleted,
        "missing": missing,
        "failed": failed,
        "protected": sorted(state.get("protected", [])),
    }


def toggle_protection(dataset: Path, state: Dict, rel_raw: str, protected: bool) -> List[str]:
    path = resolve_rel(dataset, rel_raw)
    if not path.is_file():
        raise FileNotFoundError("no such image in this dataset")
    rel = rel_key(dataset, path)
    pins = set(state.get("protected", []))
    if protected:
        pins.add(rel)
    else:
        pins.discard(rel)
    state["protected"] = sorted(pins)
    save_state(dataset, state)
    return state["protected"]


# ---------------------------------------------------------------------------
# thumbnails and previews


def _derived_lock(key: str) -> threading.Lock:
    with DERIVED_LOCKS_LOCK:
        return DERIVED_LOCKS.setdefault(key, threading.Lock())


def ensure_derived(dataset: Path, rel: str, kind: str) -> Path:
    """Create/reuse a cached JPEG. Thumbnails stay small on purpose.

    A full-size booru JPEG decodes to tens of megabytes; thirty of those as
    grid tiles is the difference between a fast phone grid and a multi-second
    stall. The browser is therefore never pointed at /image/ for a tile.
    """
    from PIL import Image, ImageOps

    source = resolve_rel(dataset, rel)
    if not source.is_file():
        raise FileNotFoundError(rel)
    safe = rel.replace("/", "__")
    destination = run_dir(dataset) / kind / f"{safe}.jpg"
    max_px = 400 if kind == "thumbs" else 1280

    lock = _derived_lock(str(destination))
    with lock:
        try:
            if destination.stat().st_mtime >= source.stat().st_mtime:
                return destination
        except OSError:
            pass

        with DERIVED_SEM:
            with Image.open(source) as img:
                img.seek(0)
                img = ImageOps.exif_transpose(img)
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                img.thumbnail((max_px, max_px), Image.LANCZOS)
                destination.parent.mkdir(parents=True, exist_ok=True)
                tmp = destination.with_suffix(".jpg.tmp")
                img.convert("RGB").save(
                    tmp, format="JPEG", quality=82, optimize=True
                )
                tmp.replace(destination)
    return destination


# ---------------------------------------------------------------------------
# scan process


class ScanJob(threading.Thread):
    """Read the worker's JSON lines and apply the result if still current."""

    daemon = True

    def __init__(
        self,
        proc: subprocess.Popen,
        dataset: Path,
        generation: int,
        config: Dict,
        seed_used: int,
        result_path: Path,
        stderr_tail: List[str],
        drain_thread: threading.Thread,
    ):
        super().__init__()
        self.proc = proc
        self.dataset = dataset
        self.generation = generation
        self.config = config
        self.seed_used = seed_used
        self.result_path = result_path
        self.stderr_tail = stderr_tail
        self.drain_thread = drain_thread

    def run(self) -> None:
        result_seen = False
        try:
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not generation_current(self.generation, self.dataset):
                    continue
                if event.get("event") == "progress":
                    set_job(
                        phase="scanning",
                        stage=str(event.get("stage", "")),
                        done=int(event.get("done", 0)),
                        total=int(event.get("total", 0)),
                        current=str(event.get("current", "")),
                        error=None,
                    )
                elif event.get("event") == "result":
                    result_seen = True
                elif event.get("event") == "error":
                    set_job(phase="error", error=str(event.get("error", "scan failed")))

            self.proc.wait()
            self.drain_thread.join(timeout=1)
            # The reader has drained stdout. Close both pipes even when the
            # generation is stale, so tests and repeated scans do not leak fds.
            _close_pipes(self.proc)
            if not generation_current(self.generation, self.dataset):
                return
            if self.stderr_tail and (
                self.config.get("verbose") or self.config.get("use_gpu")
            ):
                # The UI's "diagnostics" toggle and GPU detection write here.
                # The worker's stdout is reserved for the JSON protocol, so this
                # is the only place those messages can be seen.
                print(f"  scan log ({self.dataset.name}):", flush=True)
                for line in self.stderr_tail:
                    print(f"    {line}", flush=True)
            if self.proc.returncode != 0 or not result_seen:
                tail = "\n".join(self.stderr_tail[-8:]).strip()
                set_job(
                    phase="error",
                    stage="",
                    error=tail or f"scan worker exited with {self.proc.returncode}",
                    finished=now(),
                )
                return

            set_job(phase="loading", stage="loading", current="", error=None)
            payload = json.loads(self.result_path.read_text())
            previous = get_state(self.dataset)
            protected = previous.get("protected", []) if previous else []
            state = _state_from_payload(
                self.dataset, payload, self.config, self.seed_used, protected
            )
            pruned = prune_missing(self.dataset, state)
            save_state(self.dataset, state)
            total = job_snapshot().get("total", 0)
            set_job(
                phase="done",
                stage="",
                done=int(total),
                current="",
                error=None,
                finished=now(),
            )
            if pruned:
                print(
                    f"  pruned {len(pruned)} missing file(s) from the previous scan",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            if generation_current(self.generation, self.dataset):
                set_job(
                    phase="error",
                    stage="",
                    error=f"{type(exc).__name__}: {exc}",
                    finished=now(),
                )


def _drain(pipe, tail: List[str]) -> None:
    try:
        for line in pipe:
            line = line.rstrip()
            if line:
                tail.append(line)
                del tail[:-40]
    except Exception:  # noqa: BLE001
        pass


def _close_pipes(proc) -> None:
    """Close a worker's pipes so a finished/killed scan leaks no descriptors."""
    for stream in (proc.stdout, proc.stderr):
        try:
            if stream is not None:
                stream.close()
        except (OSError, ValueError):
            pass


def cancel_scan(mark_cancelled: bool = True) -> None:
    """Invalidate the current generation and stop its worker."""
    bump_generation()
    with CFG_LOCK:
        proc = CFG.get("scan_proc")
        CFG["scan_proc"] = None
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
    if proc is not None:
        _close_pipes(proc)
    if mark_cancelled:
        set_job(
            phase="cancelled",
            stage="",
            current="",
            error=None,
            finished=now(),
        )


def start_scan(dataset: Path, config: Dict) -> Dict:
    """Spawn one worker. Caller has already validated ``config``."""
    cancel_scan(mark_cancelled=False)
    generation = bump_generation()

    seed_used = config.get("seed")
    if seed_used is None:
        seed_used = random.randrange(1, 2**31 - 1)
    config = dict(config)
    config["seed_used"] = int(seed_used)

    directory = run_dir(dataset)
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "scan-config.json"
    result_path = directory / "scan-result.json"
    config_path.write_text(json.dumps({"directory": str(dataset), "config": config}))
    try:
        result_path.unlink()
    except OSError:
        pass

    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(WORKER),
            "--config",
            str(config_path),
            "--result",
            str(result_path),
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    tail: List[str] = []
    assert proc.stderr is not None
    drain_thread = threading.Thread(
        target=_drain, args=(proc.stderr, tail), daemon=True
    )
    drain_thread.start()

    with CFG_LOCK:
        CFG["scan_proc"] = proc
        CFG["scan_config"] = config
    set_job(
        phase="scanning",
        stage="starting",
        done=0,
        total=0,
        current="",
        error=None,
        started=now(),
        finished=None,
    )
    ScanJob(
        proc,
        dataset,
        generation,
        config,
        seed_used,
        result_path,
        tail,
        drain_thread,
    ).start()
    return config


# ---------------------------------------------------------------------------
# dataset switching


def open_dataset(dataset: Path, config: Optional[Dict] = None) -> Dict:
    dataset = dataset.expanduser().resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(f"not a directory: {dataset}")
    cancel_scan(mark_cancelled=False)
    with CFG_LOCK:
        CFG["dataset"] = dataset
        CFG["slug"] = slug_for(dataset)
        CFG["scan_config"] = None

    state = get_state(dataset)
    if state is not None:
        set_job(
            phase="done",
            stage="",
            done=0,
            total=0,
            current="",
            error=None,
            started=None,
            finished=state.get("scanned_at"),
        )
        print(f"  loaded cached scan: {state_path(dataset)}", flush=True)
    else:
        start_scan(
            dataset,
            dict(config or CFG.get("defaults", DEFAULT_CONFIG)),
        )
    return state_response()


def state_response() -> Dict:
    with CFG_LOCK:
        dataset = CFG.get("dataset")
        defaults = CFG.get("defaults", DEFAULT_CONFIG)
        scan_config = CFG.get("scan_config")
        auth = bool(CFG.get("auth"))

    state = get_state(dataset) if dataset else None
    job = job_snapshot()
    running = job.get("phase") in ("scanning", "loading")
    # While a rescan is running, report the config it was started with. A reload
    # in the middle of a scan must not snap the settings panel back to the
    # previous scan's config.
    if running and scan_config:
        current_config = scan_config
    elif state is not None:
        current_config = state.get("config")
    else:
        current_config = scan_config
    return {
        "dataset": (
            {
                "path": str(dataset),
                "name": dataset.name,
                "slug": slug_for(dataset),
            }
            if dataset
            else None
        ),
        "job": job,
        "has_results": state is not None,
        "scanned_at": state.get("scanned_at") if state else None,
        "config": current_config or defaults,
        "defaults": defaults,
        "protected_count": len(state.get("protected", [])) if state else 0,
        "pair_floor": float(
            (current_config or {}).get("pair_floor", PAIR_FLOOR)
        ),
        "auth": auth,
    }


# ---------------------------------------------------------------------------
# HTTP handler


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "similar-image-remover"

    def log_message(self, fmt, *args):  # noqa: D401 - silence per-request noise
        pass

    def handle_one_request(self) -> None:
        # Clients that reset a connection while this server is reading the next
        # request line (a closed tab, a test that aborts) otherwise print a full
        # traceback to the console. That is noise, not a server error.
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    # ---------------------------------------------------------------- helpers
    def _auth_ok(self) -> bool:
        if not CFG.get("auth"):
            return True
        header = self.headers.get("Authorization", "")
        if not header.startswith("Basic "):
            return False
        try:
            raw = base64.b64decode(header[6:], validate=True).decode("utf-8")
            username, separator, password = raw.partition(":")
            if not separator:
                return False
        except Exception:  # noqa: BLE001
            return False
        expected_user = str(CFG.get("username", "")).encode("utf-8")
        expected_pass = str(CFG.get("password", "")).encode("utf-8")
        user_ok = hmac.compare_digest(username.encode("utf-8"), expected_user)
        pass_ok = hmac.compare_digest(password.encode("utf-8"), expected_pass)
        return user_ok and pass_ok

    def _require_auth(self) -> bool:
        if self._auth_ok():
            return True
        # The browser shows its native dialog only when it sees this challenge.
        body = b"Authentication required\n"
        self.send_response(401)
        self.send_header(
            "WWW-Authenticate",
            'Basic realm="similar-image-remover", charset="UTF-8"',
        )
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if self.command == "POST":
            # The body (if any) was not read. Closing is the only safe way to
            # keep a keep-alive connection from being desynchronised.
            self.close_connection = True
            self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        time.sleep(0.35)  # make repeated guessing tedious without blocking others long
        return False

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _body(self) -> Dict:
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            raise ValueError("bad Content-Length")
        if length > 2_000_000:
            raise ValueError("request body too large")
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("expected a JSON object")
        return parsed

    def _serve_file(self, path: Path, cache: str = "no-store") -> None:
        if not path.is_file():
            return self._json({"error": "not found"}, 404)
        try:
            data = path.read_bytes()
        except OSError as exc:
            return self._json({"error": str(exc)}, 500)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _same_origin(self) -> bool:
        """Reject cross-origin writes, including simple no-cors form posts.

        The UI always sends X-Requested-With, which is not a "simple" header and
        therefore forces a CORS preflight that this server never approves. The
        Origin check covers the rest.
        """
        if self.headers.get("X-Requested-With") != "similar-image-remover":
            return False
        origin = self.headers.get("Origin")
        if origin:
            host = self.headers.get("Host", "")
            allowed = {f"http://{host}", f"https://{host}"}
            if origin not in allowed:
                return False
        return True

    # ------------------------------------------------------------------- GET
    def do_GET(self) -> None:  # noqa: N802
        if not self._require_auth():
            return
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            return self._serve_file(INDEX_HTML, cache="no-cache")

        if path == "/api/ping":
            dataset = CFG.get("dataset")
            return self._json(
                {
                    "ok": True,
                    "dataset": str(dataset) if dataset else None,
                    "auth": bool(CFG.get("auth")),
                }
            )

        if path == "/api/state":
            return self._json(state_response())

        if path == "/api/results":
            dataset = CFG.get("dataset")
            if not dataset:
                return self._json({"error": "no dataset selected"}, 409)
            state = get_state(dataset)
            if state is None:
                return self._json({"error": "no scan for this dataset yet"}, 409)
            query = urllib.parse.parse_qs(parsed.query)
            threshold = query.get("threshold", [state["config"].get("threshold", 0.9)])[0]
            try:
                threshold_value = float(threshold)
            except (TypeError, ValueError):
                return self._json({"error": "threshold must be a number"}, 400)
            payload = build_results(dataset, state, threshold_value)
            payload["job"] = job_snapshot()
            return self._json(payload)

        for prefix, kind in (
            ("/thumb/", "thumbs"),
            ("/preview/", "previews"),
            ("/image/", "image"),
        ):
            if path.startswith(prefix):
                dataset = CFG.get("dataset")
                if not dataset:
                    return self._json({"error": "no dataset selected"}, 409)
                rel = urllib.parse.unquote(path[len(prefix) :])
                try:
                    source = resolve_rel(dataset, rel)
                except ValueError as exc:
                    return self._json({"error": str(exc)}, 400)
                if not source.is_file():
                    return self._json({"error": "not found"}, 404)
                if kind == "image":
                    return self._serve_file(source, cache="private, max-age=300")
                try:
                    derived = ensure_derived(
                        dataset, rel_key(dataset, source), kind
                    )
                except Exception as exc:  # noqa: BLE001
                    return self._json({"error": f"could not render: {exc}"}, 500)
                return self._serve_file(derived, cache="private, max-age=300")

        return self._json({"error": "not found"}, 404)

    # ------------------------------------------------------------------ POST
    def do_POST(self) -> None:  # noqa: N802
        if not self._require_auth():
            return
        if not self._same_origin():
            self.close_connection = True
            return self._json({"error": "cross-origin request refused"}, 403)
        try:
            body = self._body()
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": f"bad JSON body: {exc}"}, 400)

        path = urllib.parse.urlparse(self.path).path

        if path == "/api/dataset":
            raw = str(body.get("path", "")).strip()
            if not raw:
                return self._json({"error": "enter a folder path"}, 400)
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            with CFG_LOCK:
                defaults = dict(CFG.get("defaults", DEFAULT_CONFIG))
            config = None
            if body.get("config") is not None:
                config, error = normalise_config(body.get("config"), defaults)
                if error:
                    return self._json({"error": error}, 400)
            try:
                return self._json(open_dataset(candidate, config))
            except FileNotFoundError:
                return self._json(
                    {"error": f"not a directory: {candidate.resolve()}"}, 400
                )
            except OSError as exc:
                return self._json({"error": f"cannot open folder: {exc}"}, 400)

        if path == "/api/scan":
            dataset = CFG.get("dataset")
            if not dataset:
                return self._json({"error": "choose a dataset first"}, 409)
            with CFG_LOCK:
                defaults = dict(CFG.get("defaults", DEFAULT_CONFIG))
            config, error = normalise_config(body, defaults)
            if error:
                return self._json({"error": error}, 400)
            config = start_scan(dataset, config)
            return self._json({"job": job_snapshot(), "config": config})

        if path == "/api/cancel":
            cancel_scan(mark_cancelled=True)
            return self._json({"job": job_snapshot()})

        if path == "/api/protect":
            dataset = CFG.get("dataset")
            if not dataset:
                return self._json({"error": "choose a dataset first"}, 409)
            state = get_state(dataset)
            if state is None:
                return self._json({"error": "no scan for this dataset yet"}, 409)
            rel = str(body.get("path", ""))
            try:
                pins = toggle_protection(
                    dataset, state, rel, bool(body.get("protected"))
                )
            except ValueError as exc:
                return self._json({"error": str(exc)}, 400)
            except FileNotFoundError as exc:
                return self._json({"error": str(exc)}, 404)
            return self._json({"protected": pins})

        if path == "/api/delete":
            dataset = CFG.get("dataset")
            if not dataset:
                return self._json({"error": "choose a dataset first"}, 409)
            state = get_state(dataset)
            if state is None:
                return self._json({"error": "no scan for this dataset yet"}, 409)
            raw_paths = body.get("paths", [])
            try:
                threshold = float(
                    body.get("threshold", state["config"].get("threshold", 0.9))
                )
            except (TypeError, ValueError):
                return self._json({"error": "threshold must be a number"}, 400)
            allowed, errors = validate_deletions(
                dataset, state, raw_paths, threshold
            )
            if errors:
                return self._json(
                    {"error": "refused: nothing was deleted", "details": errors},
                    400,
                )
            result = perform_deletions(dataset, state, allowed)
            return self._json(result)

        return self._json({"error": "not found"}, 404)

    # Preflight is deliberately denied: no Access-Control-Allow-* headers.
    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()


# ---------------------------------------------------------------------------
# binding and startup


def lan_addresses() -> List[str]:
    ips = set()
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("8.8.8.8", 80))  # no packet is sent for UDP connect
            address = probe.getsockname()[0]
            if not address.startswith("127."):
                ips.add(address)
        finally:
            probe.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if not address.startswith("127."):
                ips.add(address)
    except OSError:
        pass
    return sorted(ips)


def _bind(host: str, port: int):
    try:
        server = ThreadingHTTPServer((host, port), Handler)
        server.daemon_threads = True
        return server, None
    except OSError as exc:
        return None, exc


def _diagnose(host: str, port: int, last_error: Optional[OSError]) -> None:
    print("\nError: could not open a listening socket on any address.", file=sys.stderr)
    print(
        f"  tried {host}:{port} and 127.0.0.1:{port} — "
        f"{last_error or 'no socket available'}",
        file=sys.stderr,
    )
    print("\nLikely causes and fixes:", file=sys.stderr)
    print(
        "  - Another process is holding the port or the range:\n"
        "      lsof -nP -iTCP -sTCP:LISTEN",
        file=sys.stderr,
    )
    print(
        "  - File-descriptor or socket limit reached: check `ulimit -n`, close "
        "other servers, retry.",
        file=sys.stderr,
    )
    print(
        "  - macOS firewall or security software blocking Python: allow python3 "
        "in System Settings -> Network -> Firewall.",
        file=sys.stderr,
    )
    print(
        "  - A VPN or network extension is interfering: disconnect it and retry.",
        file=sys.stderr,
    )
    print(
        "  - Trying a privileged port: use a port >= 1024, or --port 0.",
        file=sys.stderr,
    )
    print(
        "  - If none of that helps, reboot to clear stale sockets and retry.",
        file=sys.stderr,
    )


def choose_server(host: str, port: int):
    """Bind the requested address, retrying auto ports and falling back to loopback."""
    loopback = host in ("127.0.0.1", "localhost", "::1")

    if port != 0:
        server, exc = _bind(host, port)
        if server is not None:
            return server, host, None
        if exc is not None and exc.errno == errno.EADDRINUSE:
            print(f"\nError: port {port} is already in use.", file=sys.stderr)
            print(
                f"Choose another port, e.g. --port {port + 1} "
                "(or --port 0 for any free port).",
                file=sys.stderr,
            )
            return None, host, exc
        if not loopback:
            fallback, fallback_exc = _bind("127.0.0.1", port)
            if fallback is not None:
                print(
                    f"\nWarning: could not bind {host}:{port} ({exc}).",
                    file=sys.stderr,
                )
                print(
                    f"Falling back to localhost only: http://127.0.0.1:{port}/",
                    file=sys.stderr,
                )
                print("Other devices will not be able to connect.", file=sys.stderr)
                return fallback, "127.0.0.1", None
            exc = fallback_exc or exc
        _diagnose(host, port, exc)
        return None, host, exc

    # Port 0: the OS selects an available port atomically. Retry anyway so an
    # address-family or race failure cannot end the run on the first try.
    last_error = None
    for _ in range(10):
        server, exc = _bind(host, 0)
        if server is not None:
            return server, host, None
        last_error = exc

    if not loopback:
        for _ in range(10):
            server, exc = _bind("127.0.0.1", 0)
            if server is not None:
                print(
                    "\nWarning: could not open a LAN-accessible socket "
                    f"({last_error}); falling back to localhost only.",
                    file=sys.stderr,
                )
                print("Other devices will not be able to connect.", file=sys.stderr)
                return server, "127.0.0.1", None
            last_error = exc

    _diagnose(host, port, last_error)
    return None, host, last_error


def run_server(
    directory: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8766,
    username: Optional[str] = None,
    password: Optional[str] = None,
    auth: bool = True,
    open_browser: bool = False,
    defaults: Optional[Dict] = None,
) -> int:
    defaults_cfg, error = normalise_config(defaults or {}, DEFAULT_CONFIG)
    if error:
        print(f"Error: invalid defaults: {error}", file=sys.stderr)
        return 2

    dataset = None
    if directory:
        dataset = Path(directory).expanduser().resolve()
        if not dataset.is_dir():
            print(f"Error: --directory is not a directory: {dataset}", file=sys.stderr)
            return 2

    with CFG_LOCK:
        CFG.clear()
        CFG.update(
            {
                "dataset": None,
                "slug": None,
                "auth": bool(auth),
                "username": username or "",
                "password": password or "",
                "defaults": defaults_cfg,
                "scan_proc": None,
                "scan_config": None,
            }
        )

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    server, bound_host, bind_error = choose_server(host, port)
    if server is None:
        return 2

    bound_port = server.server_address[1]
    print("\nSimilar Image Remover — web UI", flush=True)
    if port == 0:
        print(
            f"  port:    {bound_port} (--port 0; assigned automatically)",
            flush=True,
        )
    print(
        f"  auth:    {'username + password' if auth else 'DISABLED (--no-auth)'}",
        flush=True,
    )
    if dataset:
        print(f"  dataset: {dataset}", flush=True)
    else:
        print("  dataset: (none — choose a folder in the page)", flush=True)

    if not auth:
        print(
            "\n  WARNING: authentication is disabled (--no-auth).\n"
            "  Anyone who can reach this port can view and delete files.",
            flush=True,
        )

    local_url = f"http://127.0.0.1:{bound_port}/"
    print(f"\n  local:   {local_url}", flush=True)
    if bound_host in ("127.0.0.1", "localhost", "::1"):
        print("  network: disabled (loopback only)", flush=True)
    else:
        addresses = lan_addresses()
        if addresses:
            for address in addresses:
                print(f"  network: http://{address}:{bound_port}/", flush=True)
        else:
            print(
                "  network: no LAN address detected; other devices cannot connect",
                flush=True,
            )
    print("\n  Ctrl+C stops the server and frees the port.\n", flush=True)

    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(local_url)).start()

    previous_sigterm = None
    try:
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    except (ValueError, OSError):
        pass

    try:
        if dataset:
            open_dataset(dataset)
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping server…", flush=True)
    finally:
        cancel_scan(mark_cancelled=False)
        server.server_close()
        if previous_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, previous_sigterm)
            except (ValueError, OSError):
                pass
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Review server for similar-image-remover (standard library only)."
    )
    parser.add_argument(
        "--directory", "-d", default=None,
        help="initial folder; the page can change it later",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--no-auth", action="store_true")
    parser.add_argument("--open", dest="open_browser", action="store_true")
    args = parser.parse_args(argv)

    if not args.no_auth and not (args.username and (args.password or os.environ.get("SIR_PASSWORD"))):
        print(
            "Error: --username and --password are required (or --no-auth).",
            file=sys.stderr,
        )
        return 2
    return run_server(
        directory=args.directory,
        host=args.host,
        port=args.port,
        username=args.username,
        password=args.password or os.environ.get("SIR_PASSWORD"),
        auth=not args.no_auth,
        open_browser=args.open_browser,
    )


if __name__ == "__main__":
    sys.exit(main())
