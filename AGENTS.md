# AGENTS.md — similar-image-remover

Python tool that finds and deletes visually similar images using perceptual hashing,
with optional ORB. It has a CLI and a stdlib-HTTP web UI in `web/`. No build step, CI,
linter or formatter config. `README.md` is user-facing; `docs/reverse-proxy.md` covers
reverse proxies. This file is the working spec.

## Run

Dependencies are installed in the `python3` on PATH (3.13.11, imagehash 4.3.2,
Pillow 12.3.0, pillow-heif 1.8.1); `pip install -r requirements.txt` only if that
stops being true. Keep the requirements pins: Pillow `<13`, pillow-heif `<1.9`
(1.9 needs Pillow 13, which breaks other tools in this environment).

```bash
python3 similar_image_remover.py -d /path/to/images --hide-images      # dry run
python3 similar_image_remover.py -d /path/to/images --delete           # prompts
python3 similar_image_remover.py -d /path/to/images --delete -y        # no prompt

# web UI: auth is required unless --no-auth is explicit
python3 similar_image_remover.py --web --username me --password secret
python3 similar_image_remover.py --web --no-auth --port 0 -d /path/to/images
python3 web/server.py --username me --password secret                 # direct entry too
```

- **Always pass `--hide-images` when CLI output is captured.** The default terminal
  image escapes wreck piped or redirected output.
- `--delete` and the web UI remove files with `os.remove`: no trash, no undo.
- `-d` scans recursively. Point it at the exact dataset, not a parent of many.
- Web mode binds `0.0.0.0`, requires `--username`/`--password` (or `SIR_PASSWORD`)
  unless `--no-auth`, and prints local + LAN URLs. `--port 0` prints the assigned
  port; `--local-only` binds loopback.

## Test

```bash
python3 -m unittest test_similar_image_remover -v
python3 test_similar_image_remover.py            # equivalent
```

- Four `TestIntegration` tests skip because `test_images/` and its generator were
  removed. `OK (skipped=4)` is expected (99 of 103 run), not a regression.
- `TestWebServer` starts a real loopback server, overrides `RUNS_ROOT` with a temp
  dir, and polls `/api/state` until the scan finishes. `TestFolderPickerHelpers`
  covers counts, browsing, proxy/mobile classification and native-dialog parsing.
  Keep tests hermetic: never point them at a live `runs/`.
- Focused: `...TestWebServer`, `...TestFolderPickerHelpers`,
  `...TestResolutionPreference`.

## Architecture

- CLI: `main()` → `find_similar_images()` → `select_files_to_keep()` → preview/delete.
- Web: `web/server.py` (HTTP, auth, state, deletion policy) → `web/scan_worker.py`
  (subprocess running `find_similar_images`, JSON-lines progress) →
  `web/index.html` (settings + grouped grid). Hashing never runs in the browser.
- `select_files_to_keep()` is the only keep/delete policy. The CLI,
  `delete_similar_images()` and the web result builder all call it; do not inline a
  second selection rule.
- Keep policy: highest pixel area (`width * height`) survives; exact-resolution ties
  are random and printed. `--seed` reproduces them. The web server stores a
  `seed_used` at scan time so repeated `/api/results` calls pick the same keeper.
- `_similarity_groups()` is union-find because similarity is transitive (A~B, B~C
  groups all three); a per-key pass keeps the wrong survivor on chains.
- State: `runs/<dataset-slug>/state.json` (pair graph, config, pins, resolutions,
  sizes), `thumbs/` 400 px, `previews/` 1280 px, `deleted.log`; `runs/recent.json`
  holds shared recent folders. `runs/` is gitignored. Keys are dataset-relative POSIX.
- The folder picker is server-side: `GET /api/browse`, `POST /api/mkdir`,
  `POST /api/pick-folder`. A phone never browses its own disk; `client_context()`
  gates the native dialog.
- `/api/browse` returns a recursive image count and per-format breakdown for the
  current folder, direct counts for children, capped at 20k files / ~1.5 s with a
  truncation flag. It uses the scanner's extension set.
- UI URLs are relative (`api/state`, `thumb/...`) so a reverse proxy can strip a
  subpath such as `/similar/`. Do not add leading slashes; see `docs/reverse-proxy.md`.
- `build_results()` filters the cached pair graph at the requested threshold,
  clamped to the state's pair floor, then calls `select_files_to_keep()`;
  `validate_deletions()` re-checks every request before `perform_deletions()`.
- Auth is HTTP Basic on every route. A 401 with `WWW-Authenticate` is what triggers
  the browser dialog. Mutating POSTs also require
  `X-Requested-With: similar-image-remover` and a matching `Origin`; the server never
  sends CORS headers.

## Gotchas

- **`hash_size` and similarity are coupled.** `hash_similarity()` divides by the
  hash's real bit count (`hash.size`), not a fixed 64. `calculate_hashes` /
  `find_similar_images` default to **8**, while the CLI passes **32**; direct callers
  silently get different matches. Pass `hash_size` explicitly.
- **Formats are centralized** in `supported_image_extensions()`
  (`similar_image_remover.py`), shared by scanning and picker counts. AVIF is native
  in Pillow 12.3; HEIC/HEIF come from `pillow-heif`. RAW, SVG, PDF and video are
  intentionally unsupported.
- **The picker lists directories, never images.** Hidden dot-dirs need
  `show_hidden=1` before they can be entered or selected, but the recursive count
  still includes them because the scan does.
- **The native dialog is local-only.** `client_context()` allows it only for a
  non-mobile, effectively-local client. Loopback is a trusted proxy by default;
  `--trusted-proxy` adds others; forwarded headers from untrusted peers are ignored.
  A proxied request with no trusted client IP, and any mobile user agent, is remote.
  (The Android emulator's `10.0.2.2` arrives as loopback.) `--no-native-picker`
  disables it. One dialog at a time, 300 s timeout.
- **Do not let a phone pop a dialog on the server.** `POST /api/pick-folder`
  re-checks `client_context()` and returns 403 for remote/mobile/proxied callers.
- **New folders are created on the server.** `make_folder()` accepts one path
  segment, rejects separators/`.`/`..`/control characters (and Windows-invalid
  names), reports collisions, and clears the count cache. `--no-auth` exposes the
  folder browser as well as deletion; auth is the boundary, not a path sandbox.
- **The pair floor is 0.5 by default, not a hard limit.** A scan caches pairs down to
  its own threshold, capped at `PAIR_FLOOR` (0.9 → 0.5; 0.3 → 0.3). ORB verifies every
  cached pair at or above that floor; that is the cost of instant threshold changes.
- **Do not trust the delete request.** `validate_deletions()` refuses pinned paths,
  paths outside the current result graph, and any set that would leave a similarity
  group with no existing file. Pins are excluded from `deletable`, bulk selection and
  individual deletion; `toggle_protection()` is the only way to change them, and they
  persist in `state.json`.
- **The scan worker's stdout is a JSON protocol.** Core prints and tqdm must go to
  stderr; the worker redirects `sys.stdout` for that reason. Never print to stdout
  from code reachable by `web/scan_worker.py`.
- **Grids use cached thumbnails (400 px) and previews (1280 px), never originals.**
  A screen of originals is hundreds of MB. The UI shows “loading…” until a thumb
  paints.
- **Dataset switching invalidates the old job.** `open_dataset()` and `start_scan()`
  bump a generation and terminate the worker; `ScanJob` checks the generation so a
  stale scan cannot write into the new dataset.
- **Port handling is deliberate.** `--port 0` lets the OS choose and prints it; an
  explicit busy port exits 2 with “choose another port”; Ctrl+C/SIGTERM terminate the
  worker and free the socket.
- **`/api/dataset` accepts an optional `config`.** The UI uses it to scan a newly
  opened folder with the settings already in the panel, not the CLI defaults.
- **Do not read the `args` namespace outside `main()`.** That caused a `NameError` in
  `find_similar_images(use_orb_verification=True)` on the first ORB rejection. Pass
  `verbose` and progress explicitly.
- `similar_images` is symmetric; dedupe pairs with `tuple(sorted(...))` or every pair
  is counted twice.
- `select_files_to_keep()` takes a `report` callback (default `print`); the web
  builder passes a no-op. Progress callbacks (`stage, done, total, current`) suppress
  tqdm; stages are `hashing`, `comparing`, `orb`.

## Conventions

- Imports stdlib → third-party → local; `Optional[X]` rather than `X | None`;
  Google-style docstrings with `:param:`/`:return:`; ~100-character lines; 4 spaces.
- Tests are `unittest.TestCase` with `test_<function>_<scenario>` names, fixtures in
  `tempfile.mkdtemp()` and cleanup in `tearDown()`. `pytest` is optional, not the
  runner.
- The web UI is one vanilla HTML file: no framework, bundler or CDN. Keep the HTTP
  layer stdlib (`http.server`) so it runs under the same `python3` as the CLI.
