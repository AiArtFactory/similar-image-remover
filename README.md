# Similar Image Remover

Find and optionally delete similar/duplicate images in a directory using perceptual hashing and optional feature-based verification. The highest-resolution copy in each group of duplicates is the one that survives.

## Features

- **Cross-platform**: Works on Windows, macOS, and Linux
- **Web UI**: Optional local/LAN review page with a folder picker and image counts, an in-page browser of the server's filesystem for phones, a grouped thumbnail grid, pinning, per-file deletion and confirmed bulk deletion
- **Resolution-aware dedupe**: keeps the copy with the most pixels, and breaks exact ties randomly (visible in the preview, reproducible with `--seed`)
- **Device detection**: reports CUDA/MPS/ROCm availability, though hashing itself runs on the CPU
- **Multiple hash methods**: Choose from 7 different perceptual hashing algorithms
- **ORB feature verification**: Optional second-stage verification using computer vision features
- **Percentage-based threshold**: Specify similarity as a percentage (e.g., 90%, 95%, 100%)
- **Safe by default**: Preview mode with confirmation before deletion
- **Terminal image display**: Shows images directly in terminal (Kitty, Ghostty, iTerm2, Sixel support)
- **Progress bars**: Visual feedback during processing
- **Recursive scanning**: Processes images in all subdirectories

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Optional: GPU Acceleration

For GPU acceleration, also install PyTorch:

```bash
# For CUDA (NVIDIA)
pip install torch

# For MPS (Apple Silicon)
pip install torch

# For AMD (ROCm) - Linux only
pip install torch --index-url https://download.pytorch.org/whl/rocm5.4.2
```

## Hash Methods

The script supports 7 different perceptual hashing algorithms:

| Method | Best For | Speed | Accuracy | Description |
|--------|----------|-------|----------|-------------|
| `ahash` | Speed | ⭐⭐⭐⭐⭐ | ⭐⭐ | Average hash - simple mean brightness |
| `phash` | Balance | ⭐⭐⭐⭐ | ⭐⭐⭐ | Perceptual hash - DCT frequency based |
| `dhash` | Edges | ⭐⭐⭐⭐ | ⭐⭐⭐ | Difference hash - gradient sensitive |
| `dhash_vertical` | Vertical edges | ⭐⭐⭐⭐ | ⭐⭐⭐ | Vertical gradient differences |
| `whash` | Structure | ⭐⭐⭐ | ⭐⭐⭐⭐ | Wavelet hash - wavelet transform |
| `colorhash` | Color | ⭐⭐⭐⭐ | ⭐⭐ | Color distribution (HSV-based) |
| `crop_resistant` | Robustness | ⭐⭐ | ⭐⭐⭐⭐⭐ | Multi-segment hash - most robust to edits |

**Default**: `ahash` + `phash` (best balance of speed and accuracy)

### Choosing a Hash Method

- **Speed prioritized**: Use just `ahash` or `dhash`
- **Accuracy prioritized**: Use `crop_resistant` or combine multiple methods
- **Different images, same style**: Use `crop_resistant` or add ORB verification
- **Edge detection**: Use `dhash` or `dhash_vertical`
- **Color matching**: Use `colorhash` (catches different images with same palette)

## Usage

### Basic Usage (Dry Run / Preview Mode)

Preview what would be deleted without actually deleting:

```bash
python similar_image_remover.py -d /path/to/images
```

### Select Hash Methods

Use specific hash method(s):

```bash
# Use only dhash (faster, edge-sensitive)
python similar_image_remover.py -d /path/to/images --hash-method dhash

# Use multiple methods (more accurate, slower)
python similar_image_remover.py -d /path/to/images --hash-method dhash --hash-method whash

# Use crop-resistant hash (most robust)
python similar_image_remover.py -d /path/to/images --hash-method crop_resistant --threshold 0.85

# Adjust hash size (larger = more detail)
python similar_image_remover.py -d /path/to/images --hash-method phash --hash-size 64 --threshold 0.90
```

### Enable ORB Feature Verification

For maximum accuracy, enable ORB feature matching as a second stage:

```bash
# Basic ORB verification
python similar_image_remover.py -d /path/to/images --verify-orb

# Strict ORB verification (more matches required)
python similar_image_remover.py -d /path/to/images --verify-orb --orb-min-matches 20

# Combine with specific hash method
python similar_image_remover.py -d /path/to/images --hash-method dhash --verify-orb --threshold 0.85
```

**How ORB verification works:**
1. First stage: Hash comparison finds candidate pairs
2. Second stage: ORB extracts distinctive visual features from both images
3. If fewer than `min_matches` features match, the pair is rejected
4. This prevents false positives from images with similar style but different content

### Delete with Confirmation

Shows a preview and prompts for confirmation before deleting:

```bash
python similar_image_remover.py -d /path/to/images --delete
```

**Workflow:**
1. Scans and finds similar images
2. Displays preview (with images) showing what will be kept vs deleted
3. Prompts: "Delete X files? (y/n): "
4. Press `y` to confirm deletion, `n` to cancel

### Delete Without Confirmation (Automation)

Use `-y` or `--force` **with** `--delete` to skip the confirmation prompt (for scripts/automation):

```bash
# Short flag
python similar_image_remover.py -d /path/to/images --delete -y

# Long flag
python similar_image_remover.py -d /path/to/images --delete --force
```

**Important**: The `-y`/`--force` flag requires `--delete`. Using `-y` without `--delete` will show an error and exit without deleting anything. This prevents accidental deletions.

### Set Similarity Threshold

Control how similar images must be to be considered duplicates:

```bash
# Very strict - only identical images (100%)
python similar_image_remover.py -d /path/to/images --threshold 1.0 --delete

# Strict - 95% similarity or higher (default is 90%)
python similar_image_remover.py -d /path/to/images --threshold 0.95 --delete

# Lenient - 80% similarity or higher
python similar_image_remover.py -d /path/to/images --threshold 0.80 --delete

# Very lenient - 50% similarity or higher
python similar_image_remover.py -d /path/to/images --threshold 0.50 --delete
```

**Note**: Threshold is specified as a decimal (0.0 to 1.0) where:
- `1.0` = 100% identical (exact duplicates)
- `0.90` = 90% similar (default, catches near-duplicates)
- `0.50` = 50% similar (catches loosely similar images)

### With GPU Acceleration

```bash
python similar_image_remover.py -d /path/to/images --delete --gpu
```

### Verbose Output

```bash
python similar_image_remover.py -d /path/to/images -v
# or
python similar_image_remover.py -d /path/to/images --verbose
```

### Display Images in Terminal

Images are **shown by default** when your terminal supports it (Kitty, Ghostty, iTerm2). Use `--hide-images` to suppress or `--ascii-images` to force ASCII art display:

```bash
# Images shown by default (if terminal supports it)
python similar_image_remover.py -d /path/to/images --threshold 0.99 --delete

# Hide images (text only)
python similar_image_remover.py -d /path/to/images --delete --hide-images

# Force ASCII art display (works in any terminal)
python similar_image_remover.py -d /path/to/images --delete --ascii-images

# Control image size (width and height in terminal cells)
python similar_image_remover.py -d /path/to/images --delete --image-width 30 --image-height 15
```

**Large Image Handling:**
Images exceeding 1000 pixels in either dimension are automatically scaled down while preserving aspect ratio. This ensures:
- Images display correctly in the terminal
- No distortion or warping of the image
- Original aspect ratio is maintained
- Terminal performance is not impacted by huge images

For example, a 4000x3000 image will be scaled to 1000x750 before display.

**Important Notes:**
- **Do not pipe the output** (e.g., don't use `| head`, `| less`, `| grep`) - this prevents the image protocol from working
- Images are displayed during the preview before the confirmation prompt
- The images will be displayed as escape sequences if piped/redirected - this is expected

**Supported Terminals:**
- **Kitty** - Full native support (requires `kitten` command from kitty package)
- **Ghostty** - Full native support via Kitty graphics protocol
- **iTerm2** (macOS) - Full native image protocol support  
- **WezTerm** - Sixel support
- **MLTerm** - Sixel support
- **xterm** (with sixel patch) - Sixel support
- **Other terminals** - Falls back to ASCII art representation

**Troubleshooting:**

If images don't display:
1. Make sure you're not piping the output (don't use `| head`, `| less`, etc.)
2. For Kitty/Ghostty: Ensure `kitten` command is available (`brew install kitty`)
3. Check if your terminal supports the protocols listed above
4. Try resizing your terminal larger if images seem cut off
5. The script auto-detects terminal capability via environment variables (`TERM`, `TERM_PROGRAM`)

## Web UI

The same scan can run as a local web app with a grouped thumbnail grid. The
server uses only the Python standard library; there is no web framework and no
build step. It uses the same Pillow stack as the CLI, plus `pillow-heif` for
HEIC/HEIF images.

```bash
# local machine + LAN, protected by a browser username/password dialog
python3 similar_image_remover.py --web --username me --password secret

# start with a folder already loaded
python3 similar_image_remover.py --web --username me --password secret -d /path/to/images

# explicit opt-out for this run only
python3 similar_image_remover.py --web --no-auth --port 8799

# let the OS choose a free port
python3 similar_image_remover.py --web --username me --password secret --port 0
```

**Authentication.** Web mode requires `--username` and `--password` (or the
`SIR_PASSWORD` environment variable) unless `--no-auth` is passed. Unauthenticated
requests receive HTTP 401 plus `WWW-Authenticate`, so the browser shows its native
username/password dialog. Wrong credentials re-prompt; every route, including
images, is protected. Basic Auth over plain HTTP is not encrypted — use it on your
own Wi-Fi or pass `--local-only`, not on a public network. `--no-auth` prints a
warning and lets anyone who can reach the port view and delete files.

**Local and LAN access.** Web mode binds `0.0.0.0` by default and prints both
URLs at startup:

```
local:   http://127.0.0.1:8799/
network: http://192.168.1.23:8799/
```

Open the `local` URL on this machine, or the `network` URL on a phone/tablet on
the same Wi-Fi. If macOS asks whether to allow Python to accept connections,
allow it. `--local-only` binds `127.0.0.1` instead. `--port N` picks the port
(default 8766); `--port 0` asks the OS for a free one and prints the number it
assigned. A port that is already taken produces a short "choose another port"
message and exit code 2. Stopping the server (Ctrl+C, `SIGTERM`, or killing the
process) releases the port.

**Choosing and switching folders.** `-d` is optional in web mode. Click
**Choose folder**:

- On the machine running the server it opens Finder / Explorer / the Linux
  folder dialog. Choosing a folder starts the scan immediately; cancelling
  changes nothing.
- On a phone or another device it opens an **in-page browser of the server's
  filesystem**. Tap folders to descend, use the breadcrumbs or Back, or paste a
  path in the **Path** field and press **Go**. **Select** / **Select this
  folder** chooses the folder you are viewing. The phone never sees its own
  storage here; the listing is the server's.
- The folder list shows **folders only**, never image previews. The current
  folder shows a recursive image count and a per-format breakdown
  (`PNG 60 · JPG 8 · JPEG 1 · WebP 2 · HEIC 1`). Each child folder shows how many
  image files are directly inside it. Counts stop at 20,000 files or about 1.5
  seconds and say so when they do.
- **Show hidden** reveals dot-directories. A hidden folder must be revealed
  before it can be entered or selected. The recursive count includes hidden
  subfolders, because the scan does.
- **＋ New folder** creates a directory on the server and enters it.
- **Places** chips (Home, Desktop, Documents, Downloads, Google Drive, AI,
  Volumes, drives) and recent folders come from the server and are shared
  between the desktop and the phone. The browser remembers the show-hidden
  choice.
- If the native dialog is unavailable (headless server, no GUI, unsupported
  platform), the page falls back to the in-page browser. The 📁 button beside
  **Choose folder** opens the in-page browser even on the desktop.

There is no sandbox: the authenticated user can browse and select any readable
directory. That is the point of the feature, and why authentication is mandatory
by default.

**Reverse proxies.** The built-in server is plain HTTP + Basic Auth and has no
TLS. For HTTPS, a hostname, access from outside the LAN, or a single entry point,
put Caddy, Nginx, Tailscale Serve, or Cloudflare Tunnel in front, bind the app
with `--local-only`, and pass `--public-url`. Forwarded headers are trusted only
from loopback by default; add `--trusted-proxy CIDR` for a proxy elsewhere. The
native folder dialog is deliberately **not** offered to mobile or proxied
clients; they use the in-page browser. `--no-native-picker` disables it entirely.
See [docs/reverse-proxy.md](docs/reverse-proxy.md).

**Settings.** The left panel exposes the same scan settings as the CLI: threshold,
all seven hash methods (multi-select), hash size, ORB verification and its two
tuning values, tie-break seed, GPU detection and diagnostics. Settings that change
the hashes or verification mark the panel **Settings changed — press Scan to
apply**. The threshold is different: a scan caches pairs down to its threshold,
capped at 50%, so moving the slider regroups the existing pairs instantly and
starts no scan. Lowering it below the cached floor needs a rescan, and a rescan
at that lower threshold then caches that far. The panel shows the current floor.
A blank seed is replaced by one random seed per scan, stored with the results so
the keeper does not change on reload; enter a number to make it reproducible.

**Review and delete.** Results are grouped into connected clusters. In each group
the highest-resolution file is marked **KEEP** (the same rule and random tie-break
as the CLI). Every other file starts selected for deletion; clear a checkbox to
spare one. **Pin** makes a file immune to both bulk and individual deletion until
it is unpinned. Per-tile **✕** arms on the first click and deletes on the second
within 5 seconds. **Delete N** in a group header, or **Delete permanently** on the
bottom bar, opens a confirmation listing the files and total size. Deletion is
permanent `os.remove` — there is no trash and no undo — and the server refuses to
empty a similarity group or delete a pinned file even if a stale page asks it to.
Every removal is appended to `runs/<dataset-slug>/deleted.log`.

**Cached data.** Scan results, pins, thumbnails (400 px) and previews (1280 px)
live under `runs/<dataset-slug>/` and are gitignored. A folder that has been
scanned before opens from that cache with a **Rescan** button; files removed
outside the tool are pruned when the state loads. Thumbnails are generated on
demand and cached; the grid never downloads full-size originals.

## Command Line Arguments

| Long Option | Short | Description | Default |
|-------------|-------|-------------|---------|
| `--directory` | `-d` | Path to image directory | required for CLI; optional for `--web` |
| `--delete` | | Delete with confirmation prompt | `False` (dry run) |
| `--force` | `-y` | Skip confirmation (use with --delete) | `False` |
| `--threshold` | | Similarity threshold (0.0-1.0) | `0.90` (90%) |
| `--hash-method` | | Hash method(s) - can use multiple | `ahash, phash` |
| `--hash-size` | | Hash size (larger = more detail) | `32` |
| `--verify-orb` | | Enable ORB feature verification | `False` |
| `--orb-min-matches` | | Minimum ORB matches required | `10` |
| `--orb-threshold` | | ORB confidence threshold | `0.8` |
| `--gpu` | | Use GPU acceleration | `False` |
| `--verbose` | `-v` | Show detailed output | `False` |
| `--hide-images` | | Suppress image display | `False` (images shown) |
| `--ascii-images` | | Force ASCII art display | `False` (uses terminal protocols) |
| `--image-width` | | Max width for displayed images | Original size |
| `--image-height` | | Max height for displayed images | Original size |
| `--seed` | | Seed for reproducible tie-breaking between equal resolutions | `None` (random) |
| `--web` | | Start the browser UI instead of the CLI workflow | `False` |
| `--host` | | Web mode: bind address | `0.0.0.0` (LAN) |
| `--local-only` | | Web mode: bind 127.0.0.1 only | `False` |
| `--port` | | Web mode: port (`0` = OS-assigned, printed) | `8766` |
| `--username` | | Web mode: Basic Auth username | required unless `--no-auth` |
| `--password` | | Web mode: Basic Auth password (or `SIR_PASSWORD`) | required unless `--no-auth` |
| `--no-auth` | | Web mode: serve without a login for this run | `False` |
| `--open` | | Web mode: open the local URL in a browser | `False` |
| `--trusted-proxy` | | Web mode: proxy CIDR whose forwarded headers may be trusted (repeatable) | loopback only |
| `--no-native-picker` | | Web mode: never open Finder/Explorer/a dialog on the server | `False` |
| `--public-url` | | Web mode: external URL to print in the banner | `None` |

## How It Works

### Hash-Based Comparison (Stage 1)

1. **Hash Calculation**: Computes perceptual hash(es) for each image based on selected method(s)
2. **Similarity Calculation**: Converts hash difference to a percentage (0.0 to 1.0), measured against the hash's real bit count (64 bits at size 8, 1024 at size 32, 42 for `colorhash`)
3. **Multi-Method**: When using multiple methods, returns the maximum similarity
4. **Filtering**: Only images with similarity >= threshold proceed to Stage 2 (or are reported as similar)

### ORB Feature Verification (Stage 2, Optional)

1. **Feature Extraction**: ORB algorithm finds distinctive keypoints (corners, edges, textures)
2. **Feature Matching**: Matches keypoints between image pairs using Hamming distance
3. **Filtering**: Requires minimum number of matching features (`--orb-min-matches`)
4. **Confidence Score**: Calculates match ratio for final similarity score

### Deduplication Strategy

- Groups similar images into connected clusters (A~B and B~C form one group)
- Keeps the **highest-resolution** image in each cluster, measured in total pixels (width × height)
- Deletes the lower-resolution copies
- When two or more files in a cluster share the exact same resolution, one is kept **at random** (`--seed` makes this reproducible); the choice is printed during the preview
- Shows preview of kept vs deleted before any action

## Supported Image Formats

Formats are read through Pillow 12.3+; HEIC/HEIF use `pillow-heif` (installed
from `requirements.txt`). The scanner and the folder picker use the same runtime
extension set, so the counts do not promise a format the scan cannot open.

- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`, `.jpe`, `.jfif`)
- GIF (`.gif`)
- BMP / DIB (`.bmp`, `.dib`)
- TIFF (`.tif`, `.tiff`)
- WebP (`.webp`)
- AVIF (`.avif`, native in Pillow 12.3+)
- HEIC / HEIF (`.heic`, `.heif`, via `pillow-heif`)
- ICO (`.ico`)
- TGA (`.tga`)
- Netpbm (`.ppm`, `.pgm`, `.pbm`, `.pnm`)
- PCX, SGI, XBM, XPM, WBMP, MSP
- DDS (`.dds`)
- Photoshop composite (`.psd`)
- JPEG 2000 (`.j2k`, `.jp2`)
- EPS (`.eps`, requires Ghostscript to rasterise)

Not supported: camera RAW (CR2/NEF/ARW/DNG), SVG, PDF and video. Files that
Pillow cannot decode are skipped with a warning, as before.

## Safety Features

- **Preview before deletion**: `--delete` always shows a preview first
- **Confirmation prompt**: Requires explicit 'y' confirmation before deleting
- **Safe exit**: Ctrl+C or 'n' at prompt cancels without any changes
- **Force flag**: `-y` / `--force` available for automation (use with caution)
- **Dry run by default**: Running without `--delete` is completely safe (preview only)
- **Web pins**: a pinned file cannot be deleted, individually or in bulk, until it is unpinned
- **Web group invariant**: the server refuses any request that would delete every file in a similarity group, even from a stale page

## Safety Notes

- **Always run in dry-run mode first** to see what would be deleted
- **Back up your images** before using `--delete` or the web UI
- **Review the preview** showing kept vs deleted files before confirming
- **Web deletion is permanent too**: `os.remove`, no trash and no undo. The grid requires a second click on ✕ for one file and a confirmation listing the count, names and total size for bulk actions.
- **Web auth is only as strong as the network**: Basic Auth over plain HTTP is not encrypted. Use `--local-only` or a trusted Wi-Fi network; never expose the port to the internet.
- The script uses the full file path to handle images in subdirectories correctly
- Duplicate detection is based on visual similarity, not byte-for-byte equality
- ORB verification adds significant accuracy but increases processing time

## Testing

```bash
python3 -m unittest test_similar_image_remover -v
# equivalent:
python3 test_similar_image_remover.py
```

103 tests run. The four `TestIntegration` cases skip, because `test_images/` is
gitignored and the `create_test_images.py` generator was removed from the repository.
`OK (skipped=4)` is the expected result, not a failure. The web tests start a real
server on a free loopback port and cover Basic Auth, path traversal, the dataset
picker, folder browsing and counts, HEIC/AVIF support, mkdir, proxy/mobile client
classification, the native picker endpoint, a scan, pinning, deletion invariants
and the thumbnail/preview cache.

## Examples

```bash
# Preview exact duplicates (100% identical) with default hashes
python similar_image_remover.py -d ./photos --threshold 1.0

# Find near-duplicates using dhash (good for detecting edited images)
python similar_image_remover.py -d ./photos --hash-method dhash --threshold 0.90

# High accuracy mode: crop_resistant + ORB verification
python similar_image_remover.py -d ./photos --hash-method crop_resistant --verify-orb --threshold 0.85

# Multiple hash methods for better detection
python similar_image_remover.py -d ./photos --hash-method dhash --hash-method whash --threshold 0.88

# Find and delete exact duplicates with confirmation
python similar_image_remover.py -d ./photos --threshold 1.0 --delete

# Strict ORB mode (fewer false positives)
python similar_image_remover.py -d ./photos --verify-orb --orb-min-matches 25 --threshold 0.90

# Delete with confirmation prompt (shows images, then asks)
python similar_image_remover.py -d ./photos --threshold 0.95 --delete

# Delete without confirmation (for automation/scripts)
python similar_image_remover.py -d ./photos --threshold 0.95 --delete --force

# Same but with short flags
python similar_image_remover.py -d ./photos --threshold 0.95 --delete -y

# Fast mode for large datasets (ahash only)
python similar_image_remover.py -d ./photos --hash-method ahash --threshold 0.85

# Maximum accuracy mode (slowest)
python similar_image_remover.py -d ./photos --hash-method crop_resistant --verify-orb --threshold 0.80 -v
```

## Troubleshooting False Positives

If you're getting false positives (images marked as similar when they're not):

1. **Use ORB verification**: Add `--verify-orb` to filter out style-similar but content-different images
2. **Increase threshold**: Use `--threshold 0.95` or higher
3. **Use crop_resistant hash**: `--hash-method crop_resistant` is most robust to style similarity
4. **Combine methods**: Use multiple `--hash-method` flags
5. **Increase ORB matches**: `--orb-min-matches 20` for stricter feature matching

## Performance Tips

- **Speed**: Use `ahash` alone for fastest processing
- **Balance**: Default `ahash` + `phash` works well for most cases
- **Accuracy**: `crop_resistant` + `verify-orb` is most accurate but slowest
- **GPU**: `--gpu` detects and prints CUDA/MPS, but hashing runs on the CPU because
  `imagehash` is CPU-based — it does not currently change throughput
- **Hash size**: The CLI default is `--hash-size 32`; larger values are slower and
  more precise. Similarity is measured against the hash's real bit count, so changing
  the size changes the scale consistently instead of collapsing large hashes to 0.
  Note that the callable functions default to `8`, so pass the value explicitly when
  driving them from code
