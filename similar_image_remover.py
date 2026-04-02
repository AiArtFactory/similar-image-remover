#!/usr/bin/env python3
"""
Similar Image Remover
Find and optionally delete similar/duplicate images in a directory.

This script is cross-platform and works on:
- CPU (any platform)
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- AMD GPUs (via ROCm)

Hash methods available:
- ahash: Average hash (fast, simple)
- phash: Perceptual hash (DCT-based, good balance)
- dhash: Difference hash (gradient-based, edge sensitive)
- dhash_vertical: Vertical difference hash
- whash: Wavelet hash (good for structure)
- colorhash: Color distribution hash (HSV-based)
- crop_resistant: Crop-resistant hash (multiple segments, most robust)

Feature verification (optional second stage):
- ORB: Oriented FAST and Rotated BRIEF features for accurate verification
"""

import io
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm


def get_device():
    """Get the best available device for computation."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except ImportError:
        return "cpu"


def get_hash_function(method: str):
    """
    Get the hash function for a given method.

    :param method: Hash method name
    :return: Hash function from imagehash module
    """
    import imagehash

    hash_functions = {
        "ahash": imagehash.average_hash,
        "phash": imagehash.phash,
        "dhash": imagehash.dhash,
        "dhash_vertical": imagehash.dhash_vertical,
        "whash": imagehash.whash,
        "colorhash": imagehash.colorhash,
        "crop_resistant": imagehash.crop_resistant_hash,
    }

    if method not in hash_functions:
        raise ValueError(
            f"Unknown hash method: {method}. Available: {list(hash_functions.keys())}"
        )

    return hash_functions[method]


def calculate_single_hash(img, method: str, hash_size: int = 8) -> Any:
    """
    Calculate a single hash for an image using the specified method.

    :param img: PIL Image object
    :param method: Hash method name
    :param hash_size: Size of the hash (for applicable methods)
    :return: Hash object
    """
    hash_func = get_hash_function(method)

    # Handle special cases for different hash functions
    if method == "colorhash":
        return hash_func(img, binbits=3)
    elif method == "crop_resistant":
        return hash_func(img, min_segment_size=500, segmentation_image_size=1000)
    else:
        return hash_func(img, hash_size=hash_size)


def calculate_hashes(
    directory: str,
    hash_methods: List[str] = None,
    hash_size: int = 8,
    use_gpu: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate hashes for images in a given directory using multiple methods.

    :param directory: The path to the directory containing images.
    :param hash_methods: List of hash methods to use. Default: ["ahash", "phash"]
    :param hash_size: Size of the hash for applicable methods.
    :param use_gpu: Whether to use GPU acceleration if available.
    :return: A dictionary where keys are filenames and values are dicts of hashes.
    """
    from PIL import Image

    if hash_methods is None:
        hash_methods = ["ahash", "phash"]

    hashes = {}
    all_files = []

    # Supported image extensions
    image_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")

    # Walk through directory to get the list of all image files
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(image_extensions):
                image_path = os.path.join(dirpath, filename)
                all_files.append(image_path)

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not all_files:
        print(f"No image files found in: {directory}")
        return hashes

    device = get_device() if use_gpu else "cpu"
    if use_gpu and device != "cpu":
        print(f"Using device: {device}")

    # Use tqdm to display progress bar
    for image_path in tqdm(all_files, desc="Calculating hashes", unit="image"):
        try:
            with Image.open(image_path) as img:
                image_hashes = {}
                for method in hash_methods:
                    try:
                        image_hashes[method] = calculate_single_hash(
                            img, method, hash_size
                        )
                    except Exception as e:
                        if args and hasattr(args, "verbose") and args.verbose:
                            print(f"Warning: {method} failed for {image_path}: {e}")
                        continue

                if image_hashes:  # Only store if at least one hash succeeded
                    hashes[image_path] = image_hashes
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return hashes


def hash_similarity(hash1, hash2) -> float:
    """
    Calculate similarity between two hashes as a percentage.
    Handles both single hashes and multi-hashes (crop_resistant).

    :param hash1: First hash value (ImageHash or ImageMultiHash object)
    :param hash2: Second hash value (ImageHash or ImageMultiHash object)
    :return: Similarity percentage (0.0 to 1.0, where 1.0 = identical)
    """
    from imagehash import ImageMultiHash

    # Handle multi-hash (crop_resistant_hash returns ImageMultiHash)
    if isinstance(hash1, ImageMultiHash) and isinstance(hash2, ImageMultiHash):
        # For multi-hashes, compute similarity based on matching segments
        # This is more complex - we'll use the distance method
        try:
            max_diff = len(hash1.hash) * 64  # Approximate max distance
            diff = hash1 - hash2
            similarity = 1.0 - (diff / max_diff)
            return max(0.0, min(1.0, similarity))
        except:
            # Fallback: compare number of matching hashes
            matches = sum(1 for h1 in hash1.hash for h2 in hash2.hash if h1 == h2)
            max_possible = max(len(hash1.hash), len(hash2.hash))
            return matches / max_possible if max_possible > 0 else 0.0

    # Handle regular hashes
    max_diff = 64  # 8x8 hash = 64 bits (for most hash types)
    diff = hash1 - hash2
    similarity = 1.0 - (diff / max_diff)
    return max(0.0, min(1.0, similarity))  # Clamp between 0 and 1


def compute_combined_similarity(
    hashes1: Dict[str, Any], hashes2: Dict[str, Any]
) -> float:
    """
    Compute combined similarity across multiple hash methods.
    Returns the maximum similarity found across all common methods.

    :param hashes1: Dict of hashes for first image
    :param hashes2: Dict of hashes for second image
    :return: Maximum similarity across all methods (0.0 to 1.0)
    """
    if not hashes1 or not hashes2:
        return 0.0

    # Find common hash methods
    common_methods = set(hashes1.keys()) & set(hashes2.keys())
    if not common_methods:
        return 0.0

    similarities = []
    for method in common_methods:
        try:
            sim = hash_similarity(hashes1[method], hashes2[method])
            similarities.append(sim)
        except Exception:
            continue

    if not similarities:
        return 0.0

    # Return the maximum similarity (most lenient) or could use mean
    return max(similarities)


def verify_with_orb(
    img_path1: str, img_path2: str, min_matches: int = 10
) -> Tuple[bool, int, float]:
    """
    Verify image similarity using ORB feature matching.
    This is more accurate but slower than hashing.

    :param img_path1: Path to first image
    :param img_path2: Path to second image
    :param min_matches: Minimum number of matching features to consider similar
    :return: Tuple of (is_similar, num_matches, confidence)
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image

        # Load images
        img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            return False, 0, 0.0

        # Resize if images are too large (for performance)
        max_dim = 1024
        h1, w1 = img1.shape
        h2, w2 = img2.shape

        if max(h1, w1) > max_dim:
            scale = max_dim / max(h1, w1)
            img1 = cv2.resize(img1, None, fx=scale, fy=scale)
        if max(h2, w2) > max_dim:
            scale = max_dim / max(h2, w2)
            img2 = cv2.resize(img2, None, fx=scale, fy=scale)

        # Create ORB detector
        orb = cv2.ORB_create(nfeatures=500)

        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return False, 0, 0.0

        # Match features using BFMatcher with Hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # Sort by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Count good matches (distance < threshold)
        good_matches = [m for m in matches if m.distance < 50]
        num_good_matches = len(good_matches)

        # Calculate confidence based on match ratio
        total_keypoints = min(len(kp1), len(kp2))
        confidence = num_good_matches / total_keypoints if total_keypoints > 0 else 0.0

        # Consider similar if enough good matches
        is_similar = num_good_matches >= min_matches

        return is_similar, num_good_matches, confidence

    except ImportError:
        print("Warning: opencv-python not installed. ORB verification disabled.")
        return True, 0, 1.0  # Assume similar if can't verify
    except Exception as e:
        if args and hasattr(args, "verbose") and args.verbose:
            print(f"ORB verification error: {e}")
        return True, 0, 1.0  # Assume similar on error


def find_similar_images(
    directory: str,
    threshold: float = 0.90,
    hash_methods: List[str] = None,
    hash_size: int = 8,
    use_gpu: bool = False,
    use_orb_verification: bool = False,
    orb_min_matches: int = 10,
    orb_threshold: float = 0.8,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Find similar images based on hashes in a directory.

    :param directory: The path to the directory containing images.
    :param threshold: Similarity threshold (0.0 to 1.0, where 1.0 = identical). Default 0.90 (90%).
    :param hash_methods: List of hash methods to use. Default: ["ahash", "phash"]
    :param hash_size: Size of the hash for applicable methods.
    :param use_gpu: Whether to use GPU acceleration if available.
    :param use_orb_verification: Whether to use ORB feature matching as second stage.
    :param orb_min_matches: Minimum ORB matches to consider images similar.
    :param orb_threshold: Confidence threshold for ORB verification.
    :return: A dictionary where keys are image filenames and values are lists of (similar_image, similarity) tuples.
    """
    hashes = calculate_hashes(directory, hash_methods, hash_size, use_gpu)
    similar_images = {}
    processed_pairs = set()

    if not hashes:
        return similar_images

    hash_items = list(hashes.items())

    # First pass: find candidates using hashes
    candidates = []

    for i, (path1, hash1_dict) in enumerate(
        tqdm(hash_items, desc="Comparing images (hash)", unit="pair")
    ):
        for j, (path2, hash2_dict) in enumerate(hash_items):
            if i >= j:  # Avoid duplicate comparisons
                continue

            # Create a unique pair identifier
            pair_id = tuple(sorted([path1, path2]))
            if pair_id in processed_pairs:
                continue
            processed_pairs.add(pair_id)

            # Compute combined similarity across all hash methods
            similarity = compute_combined_similarity(hash1_dict, hash2_dict)

            if similarity >= threshold:
                candidates.append((path1, path2, similarity))

    # Second pass: ORB verification if requested
    if use_orb_verification and candidates:
        verified_similar = []
        for path1, path2, hash_sim in tqdm(
            candidates, desc="Verifying with ORB", unit="pair"
        ):
            is_similar, num_matches, confidence = verify_with_orb(
                path1, path2, orb_min_matches
            )

            if is_similar and confidence >= orb_threshold:
                # Weight the final similarity
                final_sim = (hash_sim * 0.3) + (confidence * 0.7)
                verified_similar.append((path1, path2, final_sim))
            elif args and hasattr(args, "verbose") and args.verbose:
                print(
                    f"  ORB rejected: {os.path.basename(path1)} vs {os.path.basename(path2)} "
                    f"({num_matches} matches, {confidence:.2%} confidence)"
                )
        candidates = verified_similar

    # Build the similar_images dictionary
    for path1, path2, similarity in candidates:
        if path1 not in similar_images:
            similar_images[path1] = []
        similar_images[path1].append((path2, similarity))

        # Also add reverse mapping
        if path2 not in similar_images:
            similar_images[path2] = []
        similar_images[path2].append((path1, similarity))

    return similar_images


def display_image_in_terminal(
    image_path: str,
    max_width: int = None,
    max_height: int = None,
    force_ascii: bool = False,
):
    """
    Display an image in the terminal using modern terminal image protocols.
    Supports: Kitty, iTerm2, Sixel, and basic ASCII fallback.

    :param image_path: Path to the image file
    :param max_width: Maximum width in terminal columns (None = use original)
    :param max_height: Maximum height in terminal rows (None = use original)
    :param force_ascii: If True, always use ASCII art regardless of terminal capabilities
    """
    from PIL import Image
    import shutil
    import base64
    import io
    import os

    MAX_PIXEL_DIMENSION = 1000  # Maximum pixel dimension for terminal display

    if not os.path.exists(image_path):
        print(f"  [Image not found: {image_path}]")
        return

    try:
        # Open the image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")

            # Get original dimensions
            orig_width, orig_height = img.size

            # Check if image exceeds maximum pixel dimensions
            if orig_width > MAX_PIXEL_DIMENSION or orig_height > MAX_PIXEL_DIMENSION:
                # Calculate scale factor to fit within MAX_PIXEL_DIMENSION while preserving aspect ratio
                scale = min(
                    MAX_PIXEL_DIMENSION / orig_width, MAX_PIXEL_DIMENSION / orig_height
                )
                new_width = int(orig_width * scale)
                new_height = int(orig_height * scale)
                img = img.resize((new_width, new_height), Image.LANCZOS)
                orig_width, orig_height = new_width, new_height

            # If max dimensions are specified, resize to fit
            if max_width is not None or max_height is not None:
                # Calculate aspect ratio
                aspect = orig_width / orig_height

                # Default to terminal size if not specified
                if max_width is None:
                    max_width = shutil.get_terminal_size().columns - 4
                if max_height is None:
                    max_height = 40  # Reasonable default

                # Resize to fit within max dimensions
                # Terminal cells are roughly 2:1 aspect ratio (width:height)
                term_aspect = aspect * 2

                if term_aspect > (max_width / max_height):
                    width = max_width
                    height = int(width / term_aspect)
                else:
                    height = max_height
                    width = int(height * term_aspect)

                width = max(1, min(width, max_width))
                height = max(1, min(height, max_height))

                img_resized = img.resize((width * 2, height * 2), Image.LANCZOS)
            else:
                # Use original dimensions (already checked for max size)
                img_resized = img
                width = orig_width
                height = orig_height

            # If force_ascii is set, use ASCII art regardless of terminal capabilities
            if force_ascii:
                _display_ascii_image(img, max_width or 40, max_height or 20)
            # Try Kitty graphics protocol first
            elif _supports_kitty_graphics():
                _display_kitty_image(image_path, img_resized)
            # Try Sixel
            elif _supports_sixel():
                _display_sixel_image(img_resized, width, height)
            # Try iTerm2 inline images
            elif _supports_iterm2():
                _display_iterm2_image(image_path, img_resized)
            else:
                # Fallback to simple ASCII representation
                _display_ascii_image(img, max_width or 40, max_height or 20)
    except Exception as e:
        print(f"  [Could not display image: {e}]")


def _supports_kitty_graphics() -> bool:
    """Check if terminal supports Kitty graphics protocol."""
    term = os.environ.get("TERM", "")
    term_program = os.environ.get("TERM_PROGRAM", "")

    # Check for Kitty
    if "KITTY_WINDOW_ID" in os.environ:
        return True
    if term in ("xterm-kitty", "kitty"):
        return True

    # Check for Ghostty (supports Kitty graphics protocol)
    if term == "ghostty" or term_program == "ghostty":
        return True

    return False


def _supports_sixel() -> bool:
    """Check if terminal supports Sixel."""
    import tty
    import termios
    import select

    # Check TERM for common sixel-supporting terminals
    term = os.environ.get("TERM", "")
    if "sixel" in term.lower() or "vt340" in term.lower():
        return True

    # Check for mlterm, wezterm, xterm with sixel
    if any(t in term.lower() for t in ("mlterm", "wezterm", "mintty")):
        return True

    return False


def _supports_iterm2() -> bool:
    """Check if terminal supports iTerm2 inline images."""
    term_program = os.environ.get("TERM_PROGRAM", "")
    return term_program == "iTerm.app"


def _display_kitty_image(image_path: str, img_resized):
    """Display image using Kitty graphics protocol (also works in Ghostty)."""
    import base64
    import shutil
    import subprocess

    # First try to use kitten icat if available (most reliable)
    kitten_path = shutil.which("kitten")
    if kitten_path:
        try:
            # Use kitten icat to display the image
            # Save image to temp file since kitten needs a path
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img_resized.save(tmp.name, format="PNG")
                tmp_path = tmp.name

            # Call kitten icat - DO NOT capture output so image displays
            subprocess.run(
                [kitten_path, "icat", "--align", "left", tmp_path],
                check=True,
            )

            # Clean up temp file
            os.unlink(tmp_path)
            return
        except Exception:
            # Fall through to direct protocol implementation
            pass

    # Fallback: Use Kitty graphics protocol directly
    try:
        # Save to PNG in memory
        buffer = io.BytesIO()
        img_resized.save(buffer, format="PNG")
        data = buffer.getvalue()
        data_b64 = base64.standard_b64encode(data).decode("utf-8")

        # Kitty graphics protocol escape sequence
        # Using direct transmission for simplicity
        # f=100: PNG format
        # s,v: size in pixels
        header = f"\033_Gf=100,s={img_resized.width},v={img_resized.height};"
        sys.stdout.write(header + data_b64 + "\033\\")
        sys.stdout.flush()
        print()

    except Exception as e:
        print(f"  [Kitty display failed: {e}]")


def _display_sixel_image(img_resized, width: int, height: int):
    """Display image using Sixel."""
    from PIL import Image

    try:
        # Convert to RGB
        if img_resized.mode != "RGB":
            img_resized = img_resized.convert("RGB")

        # Start Sixel
        sys.stdout.write("\033Pq")

        # Simple Sixel encoding - very basic
        pixels = img_resized.load()
        sixel_data = []

        # Color palette (simplified)
        for y in range(0, img_resized.height, 6):  # Sixel uses 6-pixel bands
            for x in range(img_resized.width):
                r, g, b = pixels[x, y] if y < img_resized.height else (0, 0, 0)
                # Map to 256 color palette
                color_idx = 16 + (r // 51) * 36 + (g // 51) * 6 + (b // 51)
                sixel_data.append(f"#{color_idx}~")  # ~ represents 6 pixels

        sys.stdout.write("".join(sixel_data))
        sys.stdout.write("\033\\")  # End Sixel
        sys.stdout.flush()
        print()
    except Exception as e:
        print(f"  [Sixel display failed: {e}]")


def _display_iterm2_image(image_path: str, img_resized):
    """Display image using iTerm2 inline image protocol."""
    import base64

    try:
        # Save resized image to memory buffer
        buffer = io.BytesIO()
        img_resized.save(buffer, format="PNG")
        data = buffer.getvalue()
        data_b64 = base64.b64encode(data).decode("utf-8")

        # iTerm2 OSC 1337 inline image protocol
        # Format: ESC ] 1337 ; File=[params]:[base64 data] BEL
        sys.stdout.write(
            f"\033]1337;File=inline=1;width={img_resized.width // 2}px;height={img_resized.height // 2}px:{data_b64}\007"
        )
        sys.stdout.flush()
        print()
    except Exception as e:
        print(f"  [iTerm2 display failed: {e}]")


def _display_ascii_image(img, width: int, height: int):
    """Fallback: display simple ASCII representation of image."""
    from PIL import Image

    try:
        # Resize to smaller size for ASCII
        ascii_width = min(width, 60)
        ascii_height = int((img.height / img.width) * ascii_width * 0.5)
        ascii_height = min(ascii_height, height * 2)

        img_small = img.resize((ascii_width, ascii_height), Image.LANCZOS)
        img_gray = img_small.convert("L")

        # ASCII characters from dark to light
        ascii_chars = "@%#*+=-:. "

        pixels = list(img_gray.getdata())
        ascii_str = ""
        for i, pixel in enumerate(pixels):
            if i > 0 and i % ascii_width == 0:
                ascii_str += "\n"
            ascii_str += ascii_chars[pixel // 32]

        print(ascii_str)
    except Exception as e:
        print(f"  [ASCII display failed: {e}]")


def delete_similar_images(
    similar_images: Dict[str, List[Tuple[str, float]]], dry_run: bool = True
) -> List[str]:
    """
    Delete similar images based on the similarity dictionary, keeping one image.

    :param similar_images: A dictionary where keys are filenames and values are lists of (similar_image, similarity) tuples.
    :param dry_run: If True, only print what would be deleted without actually deleting.
    :return: List of deleted (or would-be-deleted) file paths.
    """
    deleted_files = []
    files_to_keep = set()
    files_to_delete = set()

    # Determine which files to keep and which to delete
    for img_path, similar_list in similar_images.items():
        if img_path not in files_to_delete:
            # Keep this file, mark similar ones for deletion
            files_to_keep.add(img_path)
            for similar_img, _ in similar_list:
                if similar_img not in files_to_keep:
                    files_to_delete.add(similar_img)

    # Show which files will be kept
    print(f"\n{'Would keep' if dry_run else 'Keeping'} {len(files_to_keep)} files:")
    for file_path in sorted(files_to_keep):
        print(f"  ✓ {file_path}")

    # Show which files will be deleted
    if files_to_delete:
        print(
            f"\n{'Would delete' if dry_run else 'Deleting'} {len(files_to_delete)} files:"
        )
        for file_path in sorted(files_to_delete):
            print(f"  ✗ {file_path}")

            if not dry_run:
                try:
                    os.remove(file_path)
                    deleted_files.append(file_path)
                except FileNotFoundError:
                    print(f"  File not found: {file_path}. Skipping.")
                except PermissionError:
                    print(f"  Permission denied: {file_path}. Skipping.")
                except Exception as e:
                    print(f"  Error deleting {file_path}: {e}")
            else:
                deleted_files.append(file_path)
    else:
        print("\nNo files to delete.")

    return deleted_files


def main():
    parser = argparse.ArgumentParser(
        description="Find and optionally delete similar/duplicate images in a directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
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
        """,
    )

    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=True,
        help="Path to the directory containing images",
    )

    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete similar images with confirmation prompt (shows preview first)",
    )

    parser.add_argument(
        "-y",
        "--force",
        action="store_true",
        help="Skip confirmation prompt when deleting (use with --delete for automation)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Similarity threshold from 0.0 to 1.0 (default: 0.90, where 1.0 = identical). "
        "e.g., 0.90 means 90%% similarity or higher will be considered similar.",
    )

    parser.add_argument(
        "--hash-method",
        action="append",
        choices=[
            "ahash",
            "phash",
            "dhash",
            "dhash_vertical",
            "whash",
            "colorhash",
            "crop_resistant",
        ],
        dest="hash_methods",
        help="Hash method(s) to use. Can be specified multiple times. Default: ahash + phash",
    )

    parser.add_argument(
        "--hash-size",
        type=int,
        default=32,
        help="Size of the hash for applicable methods (default: 32). Larger = more detail but slower",
    )

    parser.add_argument(
        "--verify-orb",
        action="store_true",
        help="Use ORB feature matching as second stage verification (slower, more accurate)",
    )

    parser.add_argument(
        "--orb-min-matches",
        type=int,
        default=10,
        help="Minimum ORB feature matches for verification (default: 10)",
    )

    parser.add_argument(
        "--orb-threshold",
        type=float,
        default=0.8,
        help="ORB confidence threshold (default: 0.8, 0.0-1.0)",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration if available (requires PyTorch)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed output"
    )

    parser.add_argument(
        "--hide-images",
        action="store_true",
        help="Suppress image display in terminal (images are shown by default when terminal supports it)",
    )

    parser.add_argument(
        "--ascii-images",
        action="store_true",
        help="Force ASCII art display for images regardless of terminal capabilities",
    )

    parser.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="Maximum width for displayed images in terminal columns (default: original image width)",
    )

    parser.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="Maximum height for displayed images in terminal rows (default: original image height)",
    )

    global args
    args = parser.parse_args()

    # Validate that --force/-y requires --delete
    if args.force and not args.delete:
        print("Error: The -y/--force flag requires --delete flag to be specified.")
        print("\nUsage:")
        print("  python similar_image_remover.py -d <directory> --delete --force")
        print("  python similar_image_remover.py -d <directory> --delete -y")
        print("\nWithout --delete, the script runs in dry-run mode for safety.")
        sys.exit(1)

    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: Directory not found: {args.directory}")
        sys.exit(1)

    # Set default hash methods if not specified
    if not args.hash_methods:
        args.hash_methods = ["ahash", "phash"]

    # Check dependencies
    try:
        from imagehash import average_hash, phash
        from PIL import Image
    except ImportError as e:
        print(f"Error: Missing required dependency. {e}")
        print("Please install requirements:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    # Check for ORB dependencies if requested
    if args.verify_orb:
        try:
            import cv2
            import numpy as np
        except ImportError:
            print("Error: ORB verification requires opencv-python and numpy.")
            print("Please install: pip install opencv-python numpy")
            sys.exit(1)

    print(f"Scanning directory: {args.directory}")
    print(f"Hash methods: {', '.join(args.hash_methods)}")
    print(f"Hash size: {args.hash_size}")
    print(f"Similarity threshold: {args.threshold:.0%} ({args.threshold})")

    if args.verify_orb:
        print(
            f"ORB verification: Enabled (min_matches={args.orb_min_matches}, threshold={args.orb_threshold})"
        )

    # Determine mode
    if args.delete:
        if args.force:
            print("Mode: DELETE (forced, no confirmation)")
        else:
            print("Mode: DELETE (will prompt for confirmation)")
    else:
        print("Mode: DRY RUN (use --delete to actually delete)")

    # Find similar images
    similar_images_result = find_similar_images(
        args.directory,
        threshold=args.threshold,
        hash_methods=args.hash_methods,
        hash_size=args.hash_size,
        use_gpu=args.gpu,
        use_orb_verification=args.verify_orb,
        orb_min_matches=args.orb_min_matches,
        orb_threshold=args.orb_threshold,
    )

    if not similar_images_result:
        print("\nNo similar images found.")
        sys.exit(0)

    # Group results by similarity clusters
    print(f"\nFound {len(similar_images_result)} images with similarities.")

    # Show similar images (verbose or images enabled)
    if args.verbose or not args.hide_images:
        print("\nSimilar images found:")
        displayed_pairs = set()  # Track already displayed pairs

        for img_path, similar_list in similar_images_result.items():
            for similar_img, similarity in similar_list:
                # Create a unique pair identifier (sorted to avoid duplicates)
                pair_id = tuple(sorted([img_path, similar_img]))

                if pair_id in displayed_pairs:
                    continue  # Skip if already displayed
                displayed_pairs.add(pair_id)

                # Display this unique pair
                print(f"\n  {img_path}")

                if not args.hide_images:
                    # Display the original image
                    display_image_in_terminal(
                        img_path, args.image_width, args.image_height, args.ascii_images
                    )

                print(f"  is {similarity:.1%} similar to:")
                print(f"    - {similar_img}")

                if not args.hide_images:
                    # Display the similar image
                    display_image_in_terminal(
                        similar_img,
                        args.image_width,
                        args.image_height,
                        args.ascii_images,
                    )
                    print()  # Extra newline after image pair

    # Determine what would be kept/deleted
    deleted_files = []
    files_to_keep = set()
    files_to_delete = set()

    for img_path, similar_list in similar_images_result.items():
        if img_path not in files_to_delete:
            files_to_keep.add(img_path)
            for similar_img, _ in similar_list:
                if similar_img not in files_to_keep:
                    files_to_delete.add(similar_img)

    # Show preview
    print(f"\n{'=' * 60}")
    print(f"PREVIEW OF CHANGES:")
    print(f"{'=' * 60}")

    print(f"\nWould KEEP {len(files_to_keep)} files:")
    for file_path in sorted(files_to_keep):
        print(f"  ✓ {file_path}")

    if files_to_delete:
        print(f"\nWould DELETE {len(files_to_delete)} files:")
        for file_path in sorted(files_to_delete):
            print(f"  ✗ {file_path}")
    else:
        print("\nNo files would be deleted.")

    # Handle deletion with confirmation
    if args.delete:
        if not args.force:
            # Prompt for confirmation
            print(f"\n{'=' * 60}")
            while True:
                try:
                    response = (
                        input(f"Delete {len(files_to_delete)} files? (y/n): ")
                        .strip()
                        .lower()
                    )
                    if response == "y":
                        break
                    elif response == "n":
                        print("\nOperation cancelled. No files were deleted.")
                        sys.exit(0)
                    else:
                        print("Invalid input. Please enter 'y' or 'n'.")
                except KeyboardInterrupt:
                    print("\n\nOperation cancelled (Ctrl+C). No files were deleted.")
                    sys.exit(0)

        # Perform deletion
        print(f"\nDeleting {len(files_to_delete)} files...")
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                deleted_files.append(file_path)
                print(f"  ✗ Deleted: {file_path}")
            except FileNotFoundError:
                print(f"  ⚠ File not found: {file_path}")
            except PermissionError:
                print(f"  ⚠ Permission denied: {file_path}")
            except Exception as e:
                print(f"  ⚠ Error deleting {file_path}: {e}")

        print(f"\n{'=' * 60}")
        print(f"Deleted {len(deleted_files)} files successfully.")
        print(f"{'=' * 60}")
    else:
        print(f"\n{'=' * 60}")
        print(f"Dry run complete. Would delete {len(files_to_delete)} files.")
        print(f"Use --delete flag to actually delete (with confirmation).")
        print(f"Use --delete --force (or -y) to delete without confirmation.")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
