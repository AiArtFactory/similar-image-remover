# Similar Image Remover

Find and optionally delete similar/duplicate images using perceptual hashing with optional ORB feature verification.

## Installation

```bash
pip install -r requirements.txt
```

**Optional GPU acceleration:**
```bash
# CUDA (NVIDIA)
pip install torch

# Apple Silicon (MPS)
pip install torch

# AMD (ROCm) - Linux only
pip install torch --index-url https://download.pytorch.org/whl/rocm5.4.2
```

## Quick Start

```bash
# Preview mode (safe, no deletion)
python similar_image_remover.py -d /path/to/images

# Delete with confirmation
python similar_image_remover.py -d /path/to/images --delete

# Delete without confirmation (automation)
python similar_image_remover.py -d /path/to/images --delete --force
```

## Hash Methods

Seven perceptual hash algorithms are available:

- **ahash** - Average hash: Fast, simple mean brightness
- **phash** - Perceptual hash: DCT-based, good balance (default with ahash)
- **dhash** - Difference hash: Gradient-based, edge sensitive
- **dhash_vertical** - Vertical difference hash
- **whash** - Wavelet hash: Wavelet transform, good structure detection
- **colorhash** - Color distribution hash: HSV-based color matching
- **crop_resistant** - Crop-resistant hash: Multi-segment, most robust

**Default:** `ahash` + `phash`

## Command Line Reference

### Required Arguments

**`-d DIRECTORY, --directory DIRECTORY`** (required)
Path to the directory containing images to scan. Recursively searches subdirectories.

### Deletion Control

**`--delete`**
Actually delete similar images. Shows preview of what will be kept/deleted and prompts for confirmation before deleting anything. Dry run (preview only) is the default when this flag is not used.

**`-y, --force`**
Skip confirmation prompt when deleting. Requires `--delete` flag. Useful for automation and scripts. Using `-y` without `--delete` causes an error and exits without deleting.

### Similarity & Hash Configuration

**`--threshold THRESHOLD`** (default: 0.90)
Similarity threshold from 0.0 to 1.0 where 1.0 = identical. Images with similarity >= threshold are considered duplicates. Examples: `0.95` (strict), `0.80` (lenient), `1.0` (exact duplicates only).

**`--hash-method {ahash,phash,dhash,dhash_vertical,whash,colorhash,crop_resistant}`**
Hash method(s) to use. Can be specified multiple times for multiple hashes. Default combines ahash + phash. Multiple methods increase accuracy but slow processing.

**`--hash-size HASH_SIZE`** (default: 32)
Size of the hash for applicable methods. Larger values capture more detail but are slower. Default 32 provides good balance; lower values (8-16) are faster but less accurate.

### ORB Feature Verification

**`--verify-orb`**
Enable ORB (Oriented FAST and Rotated BRIEF) feature matching as second-stage verification. Slower but more accurate. ORB extracts distinctive visual features and verifies hash matches, preventing false positives from images with similar style but different content.

**`--orb-min-matches ORB_MIN_MATCHES`** (default: 10)
Minimum number of ORB feature matches required to consider images similar. Higher values reduce false positives but may miss valid matches.

**`--orb-threshold ORB_THRESHOLD`** (default: 0.8)
ORB confidence threshold from 0.0 to 1.0. Final similarity combines hash (30%) and ORB confidence (70%). Images below this threshold are rejected.

### Performance

**`--gpu`**
Use GPU acceleration if available. Requires PyTorch. Automatically detects CUDA (NVIDIA), MPS (Apple Silicon), or ROCm (AMD). Falls back to CPU if unavailable.

### Display & Output

**`-v, --verbose`**
Show detailed output including similarity percentages, file counts, and ORB rejection messages.

**`--hide-images`**
Suppress image display in terminal. By default, images display when terminal supports it (Kitty, Ghostty, iTerm2, Sixel). Use this for text-only output.

**`--ascii-images`**
Force ASCII art display for images regardless of terminal capabilities. Works in any terminal but lower quality than native protocols.

**`--image-width IMAGE_WIDTH`**
Maximum width for displayed images in terminal columns. Default uses original image width (capped at 1000px).

**`--image-height IMAGE_HEIGHT`**
Maximum height for displayed images in terminal rows. Default uses original image height (capped at 1000px).

## Complete Help Output

```
usage: similar_image_remover.py [-h] -d DIRECTORY [--delete] [-y]
                                [--threshold THRESHOLD]
                                [--hash-method {ahash,phash,dhash,dhash_vertical,whash,colorhash,crop_resistant}]
                                [--hash-size HASH_SIZE] [--verify-orb]
                                [--orb-min-matches ORB_MIN_MATCHES]
                                [--orb-threshold ORB_THRESHOLD] [--gpu] [-v]
                                [--hide-images] [--ascii-images]
                                [--image-width IMAGE_WIDTH]
                                [--image-height IMAGE_HEIGHT]

Find and optionally delete similar/duplicate images in a directory.

options:
  -h, --help            show this help message and exit
  -d DIRECTORY, --directory DIRECTORY
                        Path to the directory containing images
  --delete              Delete similar images with confirmation prompt (shows
                        preview first)
  -y, --force           Skip confirmation prompt when deleting (use with
                        --delete for automation)
  --threshold THRESHOLD
                        Similarity threshold from 0.0 to 1.0 (default: 0.90,
                        where 1.0 = identical). e.g., 0.90 means 90%%
                        similarity or higher will be considered similar.
  --hash-method {ahash,phash,dhash,dhash_vertical,whash,colorhash,crop_resistant}
                        Hash method(s) to use. Can be specified multiple
                        times. Default: ahash + phash
  --hash-size HASH_SIZE
                        Size of the hash for applicable methods (default: 32).
                        Larger = more detail but slower
  --verify-orb          Use ORB feature matching as second stage verification
                        (slower, more accurate)
  --orb-min-matches ORB_MIN_MATCHES
                        Minimum ORB feature matches for verification (default:
                        10)
  --orb-threshold ORB_THRESHOLD
                        ORB confidence threshold (default: 0.8, 0.0-1.0)
  --gpu                 Use GPU acceleration if available (requires PyTorch)
  -v, --verbose         Show detailed output
  --hide-images         Suppress image display in terminal (images are shown
                        by default when terminal supports it)
  --ascii-images        Force ASCII art display for images regardless of
                        terminal capabilities
  --image-width IMAGE_WIDTH
                        Maximum width for displayed images in terminal columns
                        (default: original image width)
  --image-height IMAGE_HEIGHT
                        Maximum height for displayed images in terminal rows
                        (default: original image height)

Hash Methods:
  ahash          - Average hash (fast, simple, default with phash)
  phash          - Perceptual hash (DCT-based, good balance, default with ahash)
  dhash          - Difference hash (gradient-based, edge sensitive)
  dhash_vertical - Vertical difference hash
  whash          - Wavelet hash (good for structure)
  colorhash      - Color distribution hash (HSV-based)
  crop_resistant - Crop-resistant hash (multiple segments, most robust)

Examples:
  # Default: use ahash + phash
  %(prog)s -d /path/to/images

  # Use only dhash (more edge-sensitive)
  %(prog)s -d /path/to/images --hash-method dhash

  # Use multiple methods for better accuracy
  %(prog)s -d /path/to/images --hash-method dhash --hash-method whash

  # Use crop-resistant hash (most robust)
  %(prog)s -d /path/to/images --hash-method crop_resistant --threshold 0.85

  # Enable ORB feature verification for maximum accuracy
  %(prog)s -d /path/to/images --verify-orb --threshold 0.90

  # Delete with confirmation
  %(prog)s -d /path/to/images --delete --threshold 0.95
```

## How It Works

### Stage 1: Hash Comparison
1. Calculate perceptual hashes for all images using selected method(s)
2. Compare each image pair, computing similarity (0.0 to 1.0)
3. Filter pairs above threshold as candidates
4. When multiple methods used, returns maximum similarity

### Stage 2: ORB Verification (Optional)
1. Load candidate image pairs
2. Extract ORB features (keypoints, descriptors)
3. Match features using Hamming distance
4. Calculate confidence from good matches (distance < 50)
5. Combine: final_similarity = (hash_sim * 0.3) + (orb_confidence * 0.7)
6. Filter by `--orb-threshold` and `--orb-min-matches`

### Deletion Strategy
- Keep first image in each similarity group
- Mark subsequent images for deletion
- Show preview: "Would keep X files / Would delete Y files"
- Require confirmation (unless `--force` used)

## Image Display

**Automatic Detection:**
- Kitty/Ghostty → Kitty graphics protocol
- iTerm2 → iTerm2 inline images
- Sixel terminals → Sixel graphics
- Others → ASCII art fallback

**Terminal Support:**
- **Kitty** - Full native support (requires `kitten` command)
- **Ghostty** - Full native via Kitty protocol
- **iTerm2** (macOS) - Full native support
- **WezTerm, MLTerm** - Sixel support
- **Others** - ASCII fallback

**Large Images:**
Images >1000px are auto-scaled using LANCZOS resampling before display.

## Supported Formats

PNG, JPEG/JPG, GIF, BMP, TIFF, WebP (scanned recursively)

## Dependencies

**Required:** `imagehash>=4.3.0`, `pillow>=9.0.0`, `tqdm>=4.62.0`

**Optional:** `opencv-python>=4.5.0`, `numpy>=1.20.0` (ORB), `torch>=1.12.0` (GPU)

## License

Apache License 2.0
