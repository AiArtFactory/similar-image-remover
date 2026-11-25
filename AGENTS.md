# AGENTS.md — similar-image-remover

Python tool that finds and deletes visually similar images using perceptual hashing,
with optional ORB feature verification. It has a command-line workflow and a
standard-library web UI (`web/`). No CI, no linter/formatter config, no build step.
`README.md` is user-facing; this file is the working spec.

## Run

Dependencies are already installed in the `python3` on PATH (miniconda, Python 3.13,
imagehash 4.3.2); `pip install -r requirements.txt` only if that stops being true.

```bash
python3 similar_image_remover.py -d /path/to/images --hide-images      # dry run
python3 similar_image_remover.py -d /path/to/images --delete           # prompts
python3 similar_image_remover.py -d /path/to/images --delete -y        # no prompt

# web UI: auth is required unless --no-auth is passed explicitly
python3 similar_image_remover.py --web --username me --password secret
python3 similar_image_remover.py --web --no-auth --port 0 -d /path/to/images
python3 web/server.py --username me --password secret                 # direct entry too
```

- **Always pass `--hide-images` when CLI output is captured.** Images render by
  default through Kitty/Sixel/iTerm2 escapes and wreck piped or redirected output.
- `--delete` and the web UI use `os.remove`: no trash, no undo. Dry run first.
- Scanning is recursive, so `-d` must point at the exact folder intended — pointing it
  at a parent of many datasets will delete across all of them.
- The web server binds `0.0.0.0` by default and requires `--username`/`--password`
  (or `SIR_PASSWORD`), or the explicit `--no-auth` opt-out. `--port 0` prints the
  OS-assigned port; `--local-only` binds loopback.

## Test

```bash
python3 -m unittest test_similar_image_remover -v
python3 test_similar_image_remover.py            # equivalent, also works
```

- `test_images/` is gitignored and absent, and its generator (`create_test_images.py`)
  was deleted in 862e056, so the four `TestIntegration` tests skip. 99 of 103 run.
  Skipped is the expected state, not a regression.
- `TestWebServer` starts a real server on a free loopback port, overrides
  `web.server.RUNS_ROOT` with a temp dir, and polls `/api/state` until the scan
  finishes. It covers 401 challenges, credentials, CSRF refusal, path traversal,
  dataset switching, results, pinning, deletion invariants, thumbnail/preview
  sizes, `/api/client`, folder browsing/counts, mkdir, the native-picker endpoint
  and proxy/mobile classification. `TestFolderPickerHelpers` covers the pure
  helpers: counts, hidden filtering, forwarded-header parsing, recent folders and
  native command result handling. Keep it hermetic: no live `runs/` writes.
- Narrow with `python3 -m unittest test_similar_image_remover.TestResolutionPreference -v`
  or `...TestWebServer -v`.

## Architecture

- CLI flow: `main()` (argparse) → `find_similar_images()` → `select_files_to_keep()`
  → preview and/or delete.
- Web flow: `web/server.py` (HTTP, auth, state, deletion policy) → `web/scan_worker.py`
  (subprocess: `find_similar_images` with a JSON-lines progress protocol) →
  `web/index.html` (settings + grouped grid). The browser never runs the hashing.
- `select_files_to_keep()` is the single source of truth for the keep/delete split.
  `delete_similar_images()`, `main()`, and the web result builder all call it — do not
  reintroduce inline selection logic, since removing that duplication is what commit
  e4375b2 was about.
- Keep policy: inside each group, the file with the most pixels (`width * height`)
  survives; exact-resolution ties are broken randomly, and the tie is printed so a dry
  run stays auditable. `--seed` makes tie-breaking reproducible. The web server
  generates and stores a `seed_used` at scan time when the user leaves the seed blank;
  without that, every `/api/results` request would pick a different keeper.
- `_similarity_groups()` uses union-find because similarity is transitive (A~B, B~C
  groups all three even when A≁C). A per-key pass is order-dependent and keeps the
  wrong survivor on chains.
- Web state lives in `runs/<dataset-slug>/`: `state.json` (pair graph, config, pins,
  resolutions, sizes), `thumbs/` (400 px), `previews/` (1280 px), `deleted.log`.
  `runs/recent.json` holds the shared recent folders; `runs/` is gitignored.
  Image keys are dataset-relative POSIX paths.
- The folder picker is server-side: `GET /api/browse` lists the server's direct
  child directories, `POST /api/mkdir` creates one, and `POST /api/pick-folder`
  opens the host's native dialog for a local client. The phone never browses its
  own files. `client_context()` decides whether the native dialog is allowed.
- `/api/browse` returns the current folder's recursive image count and per-format
  breakdown, plus each child's direct image count. Counts use the same extension
  set as the scanner, stop at 20,000 files / ~1.5 s, and flag truncation.
- The UI uses **relative** asset/API URLs (`api/state`, `thumb/...`) so a reverse
  proxy can mount the app under a subpath (for example `/similar/`) and strip it.
  Do not reintroduce leading slashes; the app still works at `/` and
  `/index.html`. See `docs/reverse-proxy.md`.
- The web server is the only writer of its state. `build_results()` filters the cached
  pair graph at the requested threshold (floor 0.5), groups it, and calls
  `select_files_to_keep()`; `validate_deletions()` re-checks every request against the
  current graph before `perform_deletions()` touches disk.
- Authentication is HTTP Basic Auth on every route. A missing/wrong header gets a 401
  plus `WWW-Authenticate`, which is what makes the browser show its native dialog.
  Mutating POSTs also require `X-Requested-With: similar-image-remover` and a matching
  `Origin`; the server never sends CORS headers, so a cross-origin page cannot drive
  it. POSTs rejected before the body is read close the connection.

## Gotchas

- **`hash_size` and similarity are coupled.** `hash_similarity()` divides the
  Hamming distance by the hash's real bit count (`hash.size`), not a fixed 64.
  The old divisor made every 32×32 near-duplicate score 0.0.
  `calculate_hashes`/`find_similar_images` default to **8**, while the CLI passes
  **32**; direct callers silently get different matches unless they pass the value
  explicitly.
- **Image formats are centralized.** `supported_image_extensions()` in the core is
  the one allowlist/registration point, used by both `calculate_hashes` and the
  folder picker counts. Pillow 12.3+ gives native AVIF; `pillow-heif` registers
  HEIC/HEIF. The requirements hold Pillow `<13` and pillow-heif `<1.9` because the
  newer pillow-heif requires Pillow 13, which breaks other tools in this
  environment. Camera RAW, SVG, PDF and video are intentionally unsupported.
- **The folder picker is not a file picker.** It lists directories only, never
  image thumbnails. Counts are by extension, recursive for the current folder and
  direct for child rows. Hidden dot-directories are hidden unless `show_hidden=1`;
  the recursive count still includes them because the scan does.
- **The native dialog is local-only.** `client_context()` allows it only for a
  non-mobile, effectively-local client. Loopback is a trusted proxy by default;
  `--trusted-proxy` adds others; forwarded headers from untrusted peers are not
  trusted. A request with proxy headers but no trusted client IP is remote for the
  native picker. `--no-native-picker` disables it. The Android emulator's
  `10.0.2.2` arrives as loopback, so mobile user agents are always remote here.
- **Do not let a phone pop a dialog on the Mac.** The native endpoint re-checks
  `client_context`; a remote/mobile/proxied request gets 403. One dialog at a time
  (`NATIVE_PICKER_LOCK`), 300 s timeout.
- **New folders are created on the server.** `make_folder()` validates one path
  segment, rejects separators/`.`/`..`/control characters (and Windows-invalid
  names), reports collisions, and clears the count cache.
- **`--no-auth` also exposes the filesystem browser.** The warning says so. Auth
  is the boundary; there is no path sandbox, because arbitrary paths were already
  allowed and the feature is meant to reach folders anywhere on the server.
- **The pair floor is 0.5 by default, not a hard limit.** The web worker caches
  pairs down to the scan's threshold, capped at `PAIR_FLOOR` (0.5), so a 0.9 scan
  still allows instant regrouping down to 0.5, while a scan at 0.3 caches to 0.3.
  With ORB enabled this means every pair at or above the floor is verified (slow
  but correct). The threshold slider warns when the requested value is below the
  cached floor.
- **Do not trust the delete request.** `validate_deletions()` refuses pinned paths,
  paths outside the current result graph, and any set that would leave a similarity
  group with no existing file. A stale tab must not be able to empty a group.
- **Pins are the "never delete" mechanism.** They are excluded from `deletable`, bulk
  selection, and individual deletion; `toggle_protection()` is the only way to change
  them. They persist in `state.json`.
- **The scan worker's stdout is a JSON protocol.** Any core `print`/tqdm output goes to
  stderr; the worker redirects `sys.stdout` for exactly that reason. Do not print to
  stdout in code reachable from `web/scan_worker.py`.
- **Thumbnails, not originals, feed the grid.** `/thumb/<rel>` is a cached 400 px JPEG,
  `/preview/<rel>` is 1280 px. Serving `/image/<rel>` to a tile would pull tens of MB
  per screen. The UI shows a "loading…" placeholder until the JPEG paints.
- **Dataset switching invalidates the old job.** `open_dataset()` and `start_scan()`
  bump a generation counter and terminate the old worker; `ScanJob` checks the counter
  before touching state, so a slow old scan cannot land in the new folder.
- **Port handling is deliberate.** `--port 0` retries and falls back to loopback with
  a warning if no LAN address can be bound; if loopback also fails it prints a causes
  and fixes block. An explicit port that is taken prints "choose another port" and
  exits 2 without pid hunting. Ctrl+C/SIGTERM terminate the worker and release the
  socket.
- **`/api/dataset` accepts an optional `config`.** That is how "open a new folder with
  the settings currently in the panel" starts the first scan with the user's choices
  instead of the CLI defaults.
- Do not read the `args` namespace from any function that can run outside `main()`.
  That pattern crashed `find_similar_images(use_orb_verification=True)` with
  `NameError` on the first ORB rejection. Verbosity is an explicit `verbose`
  parameter that `main()` fills from `args`; keep it that way.
- `similar_images` is symmetric: each match is stored under both paths. Naive
  iteration therefore double-counts every pair; dedupe with `tuple(sorted(...))` the
  way the display code does.
- `select_files_to_keep()` takes an optional `report` callback (default `print`). The
  web builder passes a no-op so a `/api/results` request does not spray tie messages
  into the server log; the CLI keeps its auditable dry-run output.
- Progress callbacks (`progress: (stage, done, total, current)`) suppress tqdm when
  supplied. Stages are `hashing`, `comparing`, `orb`; the web worker maps them to
  `/api/state`.

## Conventions

- Stdlib → third-party → local imports; `Optional[X]` rather than `X | None`;
  Google-style docstrings with `:param:`/`:return:`; ~100 character lines; 4 spaces.
- Tests are `unittest.TestCase` with `test_<function>_<scenario>` names, fixtures built
  in `tempfile.mkdtemp()` and removed in `tearDown()`. `pytest` is listed as optional
  and is not the runner.
- Web assets are one vanilla HTML file; no framework, bundler, or external CDN. Keep
  the server stdlib-only so it runs under the same `python3` as the CLI.
