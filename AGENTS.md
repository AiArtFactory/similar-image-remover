# AGENTS.md - Coding Guidelines for Similar Image Remover

## Project Overview

CLI tool for finding and deleting similar/duplicate images using perceptual hashing.
Cross-platform: supports CPU, CUDA (NVIDIA), MPS (Apple Silicon), AMD (ROCm).

## Build/Run/Test Commands

### Running the Tool
```bash
# Install dependencies
pip install -r requirements.txt

# Run the tool (dry-run preview mode)
python similar_image_remover.py -d /path/to/images

# Run with deletion (requires confirmation)
python similar_image_remover.py -d /path/to/images --delete
```

### Testing

**Run all tests:**
```bash
python test_similar_image_remover.py
```

**Run with unittest module (verbose):**
```bash
python -m unittest test_similar_image_remover -v
```

**Run a single test class:**
```bash
python -m unittest test_similar_image_remover.TestHashFunctions -v
```

**Run a single test method:**
```bash
python -m unittest test_similar_image_remover.TestHashFunctions.test_get_hash_function_valid_methods -v
```

**Run tests matching a pattern:**
```bash
python -m unittest discover -v -k "TestDevice"
```

### No Build Step
This is a pure Python script - no compilation or build process required.

## Code Style Guidelines

### Import Order
1. Standard library imports (io, os, sys, argparse, etc.)
2. Third-party imports (tqdm, PIL, cv2, numpy, etc.)
3. Local imports (if any)

Example:
```python
import io
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

from tqdm import tqdm
from PIL import Image
```

### Type Hints
- Use type hints for function parameters and return values
- Use `from typing import` imports at the top
- Use `Optional[Type]` for nullable values, not `Type | None`
- Use `Any` sparingly - prefer specific types

Example:
```python
def calculate_hashes(
    directory: str,
    hash_methods: List[str] = None,
    hash_size: int = 8,
    use_gpu: bool = False,
) -> Dict[str, Dict[str, Any]]:
```

### Docstrings
- Use Google-style docstrings (already established in codebase)
- Include :param and :return descriptions
- Keep descriptions concise

Example:
```python
def hash_similarity(hash1, hash2) -> float:
    """
    Calculate similarity between two hashes as a percentage.

    :param hash1: First hash value (ImageHash or ImageMultiHash object)
    :param hash2: Second hash value (ImageHash or ImageMultiHash object)
    :return: Similarity percentage (0.0 to 1.0, where 1.0 = identical)
    """
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Functions | snake_case | `calculate_hashes`, `get_device` |
| Classes | PascalCase | `TestHashFunctions` |
| Constants | UPPER_SNAKE_CASE | `MAX_PIXEL_DIMENSION = 1000` |
| Variables | snake_case | `image_path`, `hash_size` |
| Private functions | _leading_underscore | `_supports_kitty_graphics` |

### Formatting
- 4 spaces for indentation (no tabs)
- Line length: ~100 characters max (be reasonable)
- Use implicit line continuation with parentheses for long lines
- Two blank lines between top-level functions and classes
- One blank line between methods in a class

### Error Handling
- Use specific exception types when possible (FileNotFoundError, PermissionError)
- Provide informative error messages
- Use try/except blocks to handle optional dependencies gracefully

Example:
```python
try:
    import torch
    if torch.cuda.is_available():
        return "cuda"
except ImportError:
    return "cpu"
```

### Function Design
- Functions should do one thing (single responsibility)
- Keep functions reasonably short (<50 lines ideally)
- Use early returns to reduce nesting
- Use type hints for clarity

### Testing Patterns
- Use unittest.TestCase subclasses
- Test method names: `test_<function>_<scenario>`
- Use setUp/tearDown for test fixtures
- Mock external dependencies (os, sys) when appropriate
- Test classes group related functionality

Example:
```python
class TestHashFunctions(unittest.TestCase):
    """Tests for hash function selection and calculation."""

    def test_get_hash_function_valid_methods(self):
        """Test that all valid hash methods can be retrieved."""
        valid_methods = ["ahash", "phash", "dhash"]
        for method in valid_methods:
            func = sir.get_hash_function(method)
            self.assertIsNotNone(func)
```

### CLI Argument Patterns
- Use argparse with both short and long flags
- Provide sensible defaults
- Validate dependent flags (--force requires --delete)
- Show helpful error messages

## Project Structure

```
similar-image-remover/
├── similar_image_remover.py      # Main script (executable)
├── test_similar_image_remover.py # Unit tests
├── requirements.txt              # Dependencies
├── README.md                     # User documentation
└── AGENTS.md                     # This file
```

## Dependencies

**Core:** imagehash>=4.3.0, pillow>=9.0.0, tqdm>=4.62.0
**ORB:** opencv-python>=4.5.0, numpy>=1.20.0
**GPU:** torch>=1.12.0 (optional)
**Test:** pytest>=7.0.0 (optional, but unittest is standard)

## Notes for Agents

- This is a CLI tool, not a library - main() is the entry point
- Cross-platform support is important (Windows, macOS, Linux)
- GPU acceleration is optional - always have CPU fallback
- Image processing uses PIL and OpenCV - handle import errors gracefully
- Tests create temporary files - always clean up in tearDown
- Progress bars (tqdm) improve UX for long operations
