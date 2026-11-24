# AGENTS.md — similar-image-remover

Single-file Python CLI that finds and deletes visually similar images using perceptual
hashing, with optional ORB feature verification. No package, no build step, no CI, no
linter/formatter config. `README.md` is user-facing; this file is the working spec.

## Run

Dependencies are already installed in the `python3` on PATH (miniconda, Python 3.13,
imagehash 4.3.2); `pip install -r requirements.txt` only if that stops being true.

```bash
python3 similar_image_remover.py -d /path/to/images --hide-images      # dry run
python3 similar_image_remover.py -d /path/to/images --delete           # prompts
python3 similar_image_remover.py -d /path/to/images --delete -y        # no prompt
```

- **Always pass `--hide-images` when output is captured.** Images render by default
  through Kitty/Sixel/iTerm2 escapes and wreck piped or redirected output.
- `--delete` uses `os.remove`: no trash, no undo. Dry run first.
- Scanning is recursive, so `-d` must point at the exact folder intended — pointing it
  at a parent of many datasets will delete across all of them.

## Test

```bash
python3 -m unittest test_similar_image_remover -v
python3 test_similar_image_remover.py            # equivalent, also works
```

- `test_images/` is gitignored and absent, and its generator (`create_test_images.py`)
  was deleted in 862e056, so the four `TestIntegration` tests skip. 45 of 49 run.
  Skipped is the expected state, not a regression.
- Narrow with `python3 -m unittest test_similar_image_remover.TestResolutionPreference -v`.

## Architecture

- Flow: `main()` (argparse) → `find_similar_images()` → `select_files_to_keep()` →
  preview and/or delete. The CLI is the only real entrypoint; there is no library API.
- `select_files_to_keep()` is the single source of truth for the keep/delete split.
  `delete_similar_images()` and `main()` both call it — do not reintroduce inline
  selection logic, since removing that duplication is what commit e4375b2 was about.
- Keep policy: inside each group, the file with the most pixels (`width * height`)
  survives; exact-resolution ties are broken randomly, and the tie is printed so a dry
  run stays auditable. `--seed` makes tie-breaking reproducible.
- `_similarity_groups()` uses union-find because similarity is transitive (A~B, B~C
  groups all three even when A≁C). A per-key pass is order-dependent and keeps the
  wrong survivor on chains.

## Gotchas

- `hash_size` defaults to **8** in `calculate_hashes`/`find_similar_images`, but the
  CLI passes **32**. Code that calls the functions directly silently gets coarser
  hashes — and different matches — than the same options on the command line.
- Do not read the `args` namespace from any function that can run outside `main()`.
  That pattern crashed `find_similar_images(use_orb_verification=True)` with
  `NameError` on the first ORB rejection. Verbosity is now an explicit `verbose`
  parameter that `main()` fills from `args`; keep it that way.
- `similar_images` is symmetric: each match is stored under both paths. Naive
  iteration therefore double-counts every pair; dedupe with `tuple(sorted(...))` the
  way the display code does.

## Conventions

- Stdlib → third-party → local imports; `Optional[X]` rather than `X | None`;
  Google-style docstrings with `:param:`/`:return:`; ~100 character lines; 4 spaces.
- Tests are `unittest.TestCase` with `test_<function>_<scenario>` names, fixtures built
  in `tempfile.mkdtemp()` and removed in `tearDown()`. `pytest` is listed as optional
  and is not the runner.
