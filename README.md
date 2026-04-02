# Similar Image Remover

Find and optionally delete similar/duplicate images in a directory using perceptual hashing and optional feature-based verification.

## Features

- **Cross-platform**: Works on Windows, macOS, and Linux
- **Multi-device support**: CPU, CUDA (NVIDIA), MPS (Apple Silicon), AMD (ROCm)
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
Images exceeding 2000 pixels in either dimension are automatically scaled down while preserving aspect ratio. This ensures:
- Images display correctly in the terminal
- No distortion or warping of the image
- Original aspect ratio is maintained
- Terminal performance is not impacted by huge images

For example, a 4000x3000 image will be scaled to 2000x1500 before display.

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

## Command Line Arguments

| Long Option | Short | Description | Default |
|-------------|-------|-------------|---------|
| `--directory` | `-d` | Path to image directory | (required) |
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

## How It Works

### Hash-Based Comparison (Stage 1)

1. **Hash Calculation**: Computes perceptual hash(es) for each image based on selected method(s)
2. **Similarity Calculation**: Converts hash difference to a percentage (0.0 to 1.0)
3. **Multi-Method**: When using multiple methods, returns the maximum similarity
4. **Filtering**: Only images with similarity >= threshold proceed to Stage 2 (or are reported as similar)

### ORB Feature Verification (Stage 2, Optional)

1. **Feature Extraction**: ORB algorithm finds distinctive keypoints (corners, edges, textures)
2. **Feature Matching**: Matches keypoints between image pairs using Hamming distance
3. **Filtering**: Requires minimum number of matching features (`--orb-min-matches`)
4. **Confidence Score**: Calculates match ratio for final similarity score

### Deduplication Strategy

- Keeps the first image in each similar group
- Marks subsequent images for deletion
- Shows preview of kept vs deleted before any action

## Supported Image Formats

- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`)
- GIF (`.gif`)
- BMP (`.bmp`)
- TIFF (`.tiff`)
- WebP (`.webp`)

## Safety Features

- **Preview before deletion**: `--delete` always shows a preview first
- **Confirmation prompt**: Requires explicit 'y' confirmation before deleting
- **Safe exit**: Ctrl+C or 'n' at prompt cancels without any changes
- **Force flag**: `-y` / `--force` available for automation (use with caution)
- **Dry run by default**: Running without `--delete` is completely safe (preview only)

## Safety Notes

- **Always run in dry-run mode first** to see what would be deleted
- **Back up your images** before using `--delete`
- **Review the preview** showing kept vs deleted files before confirming
- The script uses the full file path to handle images in subdirectories correctly
- Duplicate detection is based on visual similarity, not byte-for-byte equality
- ORB verification adds significant accuracy but increases processing time

## Testing

Run the unit tests:

```bash
python test_similar_image_remover.py
```

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
- **GPU**: `--gpu` flag can speed up processing on supported hardware
- **Hash size**: Default `--hash-size 8` is usually sufficient; larger sizes are slower

## License

Apache License 2.0
