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
  was deleted in 862e056, so the four `TestIntegration` tests skip. 74 of 78 run.
  Skipped is the expected state, not a regression.
- `TestWebServer` starts a real server on a free loopback port, overrides
  `web.server.RUNS_ROOT` with a temp dir, and polls `/api/state` until the scan
  finishes. It covers 401 challenges, credentials, CSRF refusal, path traversal,
  dataset switching, results, pinning, deletion invariants, and thumbnail/preview
  sizes. Keep it hermetic: no live `runs/` writes.
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
  `runs/` is gitignored. Image keys are dataset-relative POSIX paths.
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

- **`hash_size` and similarity are coupled.** `hash_similarity()` divides the Hamming
  distance by the hash's real bit count (`hash.size`), not a fixed 64. The old divisor
  made every 32×32 near-duplicate score 0.0. `calculate_hashes`/`find_similar_images`
  default to **8**, while the CLI passes **32**; direct callers silently get different
  matches unless they pass the value explicitly.
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
