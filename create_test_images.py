#!/usr/bin/env python3
"""
Create test images for unit testing.
Generates:
- Identical images
- Visually similar images (slight modifications)
- Different images
"""

import os
from PIL import Image, ImageDraw
import random


def create_solid_image(path, color, size=(100, 100)):
    """Create a solid color image."""
    img = Image.new("RGB", size, color)
    img.save(path)


def create_gradient_image(path, start_color, end_color, size=(100, 100)):
    """Create a gradient image."""
    img = Image.new("RGB", size)
    draw = ImageDraw.Draw(img)
    for y in range(size[1]):
        r = int(start_color[0] + (end_color[0] - start_color[0]) * y / size[1])
        g = int(start_color[1] + (end_color[1] - start_color[1]) * y / size[1])
        b = int(start_color[2] + (end_color[2] - start_color[2]) * y / size[1])
        draw.line([(0, y), (size[0], y)], fill=(r, g, b))
    img.save(path)


def create_pattern_image(path, size=(100, 100)):
    """Create a patterned image."""
    img = Image.new("RGB", size)
    draw = ImageDraw.Draw(img)
    for i in range(0, size[0], 10):
        draw.rectangle([i, 0, i + 5, size[1]], fill=(i % 255, 100, 150))
    img.save(path)


def create_test_images(base_dir="test_images"):
    """Create a complete set of test images."""
    os.makedirs(base_dir, exist_ok=True)

    # Create subdirectory for nested testing
    subdir = os.path.join(base_dir, "subdir")
    os.makedirs(subdir, exist_ok=True)

    # 1. Identical images (copy of same content)
    print("Creating identical images...")
    create_solid_image(os.path.join(base_dir, "identical1.png"), (255, 0, 0))
    create_solid_image(os.path.join(base_dir, "identical2.png"), (255, 0, 0))
    create_solid_image(
        os.path.join(subdir, "identical3.png"), (255, 0, 0)
    )  # Same in subdir

    # 2. Visually similar images (slight color variations)
    print("Creating similar images...")
    create_solid_image(os.path.join(base_dir, "similar1.png"), (255, 0, 0))
    create_solid_image(
        os.path.join(base_dir, "similar2.png"), (254, 1, 0)
    )  # Almost identical
    create_solid_image(
        os.path.join(base_dir, "similar3.png"), (253, 2, 1)
    )  # Almost identical

    # 3. Different images
    print("Creating different images...")
    create_solid_image(os.path.join(base_dir, "different1.png"), (0, 255, 0))
    create_solid_image(os.path.join(base_dir, "different2.png"), (0, 0, 255))
    create_gradient_image(
        os.path.join(base_dir, "gradient.png"), (0, 0, 0), (255, 255, 255)
    )
    create_pattern_image(os.path.join(base_dir, "pattern.png"))

    # 4. Different formats
    print("Creating different format images...")
    img = Image.new("RGB", (100, 100), (128, 128, 128))
    img.save(os.path.join(base_dir, "gray.jpg"), quality=95)
    img.save(os.path.join(base_dir, "gray.gif"))
    img.save(os.path.join(base_dir, "gray.bmp"))

    # 5. Edge cases
    print("Creating edge case images...")
    # Very small image
    create_solid_image(os.path.join(base_dir, "tiny.png"), (255, 255, 0), size=(8, 8))
    # Large image
    create_solid_image(
        os.path.join(base_dir, "large.png"), (128, 128, 128), size=(1000, 1000)
    )

    print(f"\nTest images created in: {base_dir}")
    print(
        f"  - Identical images: identical1.png, identical2.png, subdir/identical3.png"
    )
    print(f"  - Similar images: similar1.png, similar2.png, similar3.png")
    print(
        f"  - Different images: different1.png, different2.png, gradient.png, pattern.png"
    )
    print(f"  - Different formats: gray.jpg, gray.gif, gray.bmp")
    print(f"  - Edge cases: tiny.png, large.png")


if __name__ == "__main__":
    create_test_images()
