#!/usr/bin/env python3
"""
Comprehensive unit tests for similar_image_remover.py
"""

import unittest
import os
import sys
import tempfile
import shutil
import json
import threading
import time
import base64
import http.client
import urllib.request
import urllib.error
from pathlib import Path
from unittest import mock
from PIL import Image

# Import the module under test
import similar_image_remover as sir


class TestDeviceDetection(unittest.TestCase):
    """Tests for device detection functionality."""

    def test_get_device_returns_string(self):
        """Test that get_device returns a string."""
        device = sir.get_device()
        self.assertIsInstance(device, str)

    def test_get_device_returns_valid_value(self):
        """Test that get_device returns a valid device string."""
        device = sir.get_device()
        valid_devices = ["cpu", "cuda", "mps", "rocm"]
        self.assertIn(device, valid_devices)


class TestHashFunctions(unittest.TestCase):
    """Tests for hash function selection and calculation."""

    def test_get_hash_function_valid_methods(self):
        """Test that all valid hash methods can be retrieved."""
        valid_methods = [
            "ahash",
            "phash",
            "dhash",
            "dhash_vertical",
            "whash",
            "colorhash",
            "crop_resistant",
        ]
        for method in valid_methods:
            func = sir.get_hash_function(method)
            self.assertIsNotNone(func)

    def test_get_hash_function_invalid_method(self):
        """Test that invalid method raises ValueError."""
        with self.assertRaises(ValueError):
            sir.get_hash_function("invalid_method")

    def test_get_hash_function_unknown_method(self):
        """Test that unknown method raises ValueError."""
        with self.assertRaises(ValueError):
            sir.get_hash_function("unknown")


class TestHashCalculation(unittest.TestCase):
    """Tests for hash calculation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()

        # Create test images
        img1 = Image.new("RGB", (100, 100), (255, 0, 0))
        img1.save(os.path.join(self.test_dir, "test1.png"))

        img2 = Image.new("RGB", (100, 100), (255, 0, 0))
        img2.save(os.path.join(self.test_dir, "test2.png"))  # Identical

        img3 = Image.new("RGB", (100, 100), (0, 255, 0))
        img3.save(os.path.join(self.test_dir, "test3.png"))  # Different

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def test_calculate_hashes_returns_dict(self):
        """Test that calculate_hashes returns a dictionary."""
        hashes = sir.calculate_hashes(self.test_dir)
        self.assertIsInstance(hashes, dict)

    def test_calculate_hashes_finds_all_images(self):
        """Test that all images are found."""
        hashes = sir.calculate_hashes(self.test_dir)
        self.assertEqual(len(hashes), 3)

    def test_calculate_hashes_returns_dict_values(self):
        """Test that hash values are dictionaries of hashes."""
        hashes = sir.calculate_hashes(self.test_dir)
        for path, hash_dict in hashes.items():
            self.assertIsInstance(hash_dict, dict)

    def test_calculate_hashes_single_method(self):
        """Test calculating hashes with a single method."""
        hashes = sir.calculate_hashes(self.test_dir, hash_methods=["ahash"])
        for path, hash_dict in hashes.items():
            self.assertIn("ahash", hash_dict)
            self.assertEqual(len(hash_dict), 1)

    def test_calculate_hashes_multiple_methods(self):
        """Test calculating hashes with multiple methods."""
        hashes = sir.calculate_hashes(self.test_dir, hash_methods=["ahash", "phash"])
        for path, hash_dict in hashes.items():
            self.assertIn("ahash", hash_dict)
            self.assertIn("phash", hash_dict)

    def test_calculate_hashes_empty_directory(self):
        """Test handling of empty directory."""
        empty_dir = tempfile.mkdtemp()
        hashes = sir.calculate_hashes(empty_dir)
        self.assertEqual(len(hashes), 0)
        shutil.rmtree(empty_dir)

    def test_calculate_hashes_nonexistent_directory(self):
        """Test handling of non-existent directory."""
        with self.assertRaises((FileNotFoundError, OSError)):
            sir.calculate_hashes("/nonexistent/path/12345")

    def test_calculate_hashes_different_hash_sizes(self):
        """Test hash calculation with different hash sizes."""
        hashes_8 = sir.calculate_hashes(
            self.test_dir, hash_methods=["ahash"], hash_size=8
        )
        hashes_16 = sir.calculate_hashes(
            self.test_dir, hash_methods=["ahash"], hash_size=16
        )

        # Both should return hashes
        self.assertEqual(len(hashes_8), 3)
        self.assertEqual(len(hashes_16), 3)


class TestHashSimilarity(unittest.TestCase):
    """Tests for hash similarity calculation."""

    def test_hash_similarity_identical(self):
        """Test that identical hashes return 1.0."""
        from imagehash import ImageHash
        import numpy as np

        hash1 = ImageHash(np.zeros((8, 8), dtype=bool))
        hash2 = ImageHash(np.zeros((8, 8), dtype=bool))

        sim = sir.hash_similarity(hash1, hash2)
        self.assertEqual(sim, 1.0)

    def test_hash_similarity_completely_different(self):
        """Test that completely different hashes return 0.0."""
        from imagehash import ImageHash
        import numpy as np

        hash1 = ImageHash(np.zeros((8, 8), dtype=bool))
        hash2 = ImageHash(np.ones((8, 8), dtype=bool))

        sim = sir.hash_similarity(hash1, hash2)
        self.assertEqual(sim, 0.0)

    def test_hash_similarity_partial(self):
        """Test partial similarity."""
        from imagehash import ImageHash
        import numpy as np

        hash1 = ImageHash(np.zeros((8, 8), dtype=bool))
        arr = np.zeros((8, 8), dtype=bool)
        arr[0:4, :] = True
        hash2 = ImageHash(arr)

        sim = sir.hash_similarity(hash1, hash2)
        self.assertAlmostEqual(sim, 0.5, places=2)

    def test_compute_combined_similarity_same_hashes(self):
        """Test combined similarity with identical hash dicts."""
        from imagehash import ImageHash
        import numpy as np

        hash_val = ImageHash(np.zeros((8, 8), dtype=bool))
        hashes1 = {"ahash": hash_val, "phash": hash_val}
        hashes2 = {"ahash": hash_val, "phash": hash_val}

        sim = sir.compute_combined_similarity(hashes1, hashes2)
        self.assertEqual(sim, 1.0)

    def test_compute_combined_similarity_partial(self):
        """Test combined similarity with partial match."""
        from imagehash import ImageHash
        import numpy as np

        hash1 = ImageHash(np.zeros((8, 8), dtype=bool))
        hash2 = ImageHash(np.ones((8, 8), dtype=bool))

        # Only ahash matches
        hashes1 = {"ahash": hash1}
        hashes2 = {"ahash": hash1, "phash": hash2}

        sim = sir.compute_combined_similarity(hashes1, hashes2)
        self.assertEqual(sim, 1.0)  # Only compares common methods

    def test_compute_combined_similarity_no_common(self):
        """Test combined similarity with no common methods."""
        from imagehash import ImageHash
        import numpy as np

        hash1 = ImageHash(np.zeros((8, 8), dtype=bool))
        hash2 = ImageHash(np.zeros((8, 8), dtype=bool))

        hashes1 = {"ahash": hash1}
        hashes2 = {"phash": hash2}

        sim = sir.compute_combined_similarity(hashes1, hashes2)
        self.assertEqual(sim, 0.0)


class TestORBVerification(unittest.TestCase):
    """Tests for ORB feature verification."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()

        # Create test images
        img1 = Image.new("RGB", (100, 100), (255, 0, 0))
        img1.save(os.path.join(self.test_dir, "test1.png"))

        img2 = Image.new("RGB", (100, 100), (255, 0, 0))  # Identical
        img2.save(os.path.join(self.test_dir, "test2.png"))

        img3 = Image.new("RGB", (100, 100), (0, 255, 0))  # Different
        img3.save(os.path.join(self.test_dir, "test3.png"))

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def test_verify_with_orb_identical_images(self):
        """Test ORB verification with images that have features."""
        import numpy as np

        # Create images with actual features (corners, edges) for ORB to detect
        img1 = Image.fromarray(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8))
        img1.save(os.path.join(self.test_dir, "orb_test1.png"))

        img2 = img1.copy()  # Identical copy
        img2.save(os.path.join(self.test_dir, "orb_test2.png"))

        img1_path = os.path.join(self.test_dir, "orb_test1.png")
        img2_path = os.path.join(self.test_dir, "orb_test2.png")

        is_similar, num_matches, confidence = sir.verify_with_orb(img1_path, img2_path)

        # Images with features should have matches
        self.assertGreaterEqual(num_matches, 0)
        self.assertGreaterEqual(confidence, 0)

    def test_verify_with_orb_different_images(self):
        """Test ORB verification with different images."""
        img1_path = os.path.join(self.test_dir, "test1.png")
        img3_path = os.path.join(self.test_dir, "test3.png")

        is_similar, num_matches, confidence = sir.verify_with_orb(
            img1_path, img3_path, min_matches=50
        )

        # Different images should have few/no matches with high threshold
        self.assertFalse(is_similar)

    def test_verify_with_orb_nonexistent_file(self):
        """Test ORB verification with non-existent file."""
        is_similar, num_matches, confidence = sir.verify_with_orb(
            "/nonexistent1.png", "/nonexistent2.png"
        )

        self.assertFalse(is_similar)
        self.assertEqual(num_matches, 0)
        self.assertEqual(confidence, 0.0)


class TestFindSimilarImages(unittest.TestCase):
    """Tests for finding similar images."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()

        # Create similar images
        img1 = Image.new("RGB", (100, 100), (255, 0, 0))
        img1.save(os.path.join(self.test_dir, "similar1.png"))

        img2 = Image.new("RGB", (100, 100), (255, 0, 0))
        img2.save(os.path.join(self.test_dir, "similar2.png"))  # Identical

        img3 = Image.new("RGB", (100, 100), (254, 1, 0))
        img3.save(os.path.join(self.test_dir, "similar3.png"))  # Very similar

        # Create different image
        img4 = Image.new("RGB", (100, 100), (0, 255, 0))
        img4.save(os.path.join(self.test_dir, "different.png"))

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def test_find_similar_images_returns_dict(self):
        """Test that find_similar_images returns a dictionary."""
        similar = sir.find_similar_images(self.test_dir, threshold=0.90)
        self.assertIsInstance(similar, dict)

    def test_find_similar_images_detects_identical(self):
        """Test that identical images are detected."""
        similar = sir.find_similar_images(self.test_dir, threshold=1.0)
        # At least the identical images should be found (100% identical)
        self.assertGreater(len(similar), 0)

    def test_find_similar_images_threshold_behavior(self):
        """Test that threshold affects detection."""
        # High threshold (strict)
        similar_strict = sir.find_similar_images(self.test_dir, threshold=0.99)
        # Low threshold (lenient)
        similar_lenient = sir.find_similar_images(self.test_dir, threshold=0.50)

        # Lenient threshold should find at least as many as strict threshold
        self.assertGreaterEqual(len(similar_lenient), len(similar_strict))

    def test_find_similar_images_empty_directory(self):
        """Test handling of empty directory."""
        empty_dir = tempfile.mkdtemp()
        similar = sir.find_similar_images(empty_dir)
        self.assertEqual(len(similar), 0)
        shutil.rmtree(empty_dir)

    def test_find_similar_images_with_dhash(self):
        """Test finding similar images with dhash method."""
        similar = sir.find_similar_images(
            self.test_dir, threshold=0.90, hash_methods=["dhash"]
        )
        self.assertIsInstance(similar, dict)

    def test_find_similar_images_with_crop_resistant(self):
        """Test finding similar images with crop_resistant method."""
        similar = sir.find_similar_images(
            self.test_dir, threshold=0.80, hash_methods=["crop_resistant"]
        )
        self.assertIsInstance(similar, dict)


class TestDeleteSimilarImages(unittest.TestCase):
    """Tests for deleting similar images."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()

        # Create test files
        self.file1 = os.path.join(self.test_dir, "file1.png")
        self.file2 = os.path.join(self.test_dir, "file2.png")
        self.file3 = os.path.join(self.test_dir, "file3.png")

        img = Image.new("RGB", (100, 100), (255, 0, 0))
        img.save(self.file2)
        img.save(self.file3)

        # file1 is the highest resolution in the group, so it must survive
        Image.new("RGB", (200, 200), (255, 0, 0)).save(self.file1)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def test_delete_dry_run_preserves_files(self):
        """Test that dry run doesn't delete files."""
        similar_images = {self.file1: [(self.file2, 0.95), (self.file3, 0.90)]}

        deleted = sir.delete_similar_images(similar_images, dry_run=True)

        # Files should still exist
        self.assertTrue(os.path.exists(self.file1))
        self.assertTrue(os.path.exists(self.file2))
        self.assertTrue(os.path.exists(self.file3))

        # But should be listed as to-delete
        self.assertIn(self.file2, deleted)
        self.assertIn(self.file3, deleted)

    def test_delete_actually_deletes(self):
        """Test that delete actually removes files."""
        similar_images = {self.file1: [(self.file2, 0.95), (self.file3, 0.90)]}

        deleted = sir.delete_similar_images(similar_images, dry_run=False)

        # Original should remain
        self.assertTrue(os.path.exists(self.file1))
        # Similar files should be deleted
        self.assertFalse(os.path.exists(self.file2))
        self.assertFalse(os.path.exists(self.file3))


class TestResolutionPreference(unittest.TestCase):
    """Tests for keeping the highest-resolution image in each group."""

    def setUp(self):
        """Set up an empty temporary directory."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.test_dir)

    def _make_image(self, name, width, height):
        """Create an image of the given size and return its path."""
        path = os.path.join(self.test_dir, name)
        Image.new("RGB", (width, height), (10, 20, 30)).save(path)
        return path

    def test_get_image_resolution_reads_dimensions(self):
        """Resolution is read from the file header."""
        path = self._make_image("a.png", 640, 480)
        self.assertEqual(sir.get_image_resolution(path), (640, 480))

    def test_get_image_resolution_missing_file(self):
        """An unreadable file reports (0, 0) instead of raising."""
        self.assertEqual(
            sir.get_image_resolution(os.path.join(self.test_dir, "nope.png")), (0, 0)
        )

    def test_keeps_highest_resolution(self):
        """The largest image in a group survives."""
        small = self._make_image("small.png", 800, 533)
        medium = self._make_image("medium.png", 1200, 800)
        large = self._make_image("large.png", 2400, 1600)

        similar = {small: [(medium, 0.95), (large, 0.92)]}
        keep, delete = sir.select_files_to_keep(similar)

        self.assertEqual(keep, [large])
        self.assertCountEqual(delete, [small, medium])

    def test_keeps_largest_across_transitive_chain(self):
        """A chain A~B, B~C is treated as one group; the largest wins."""
        a = self._make_image("a.png", 2000, 2000)
        b = self._make_image("b.png", 1000, 1000)
        c = self._make_image("c.png", 500, 500)

        # A never matched C directly, only through B
        similar = {a: [(b, 0.93)], b: [(a, 0.93), (c, 0.91)]}
        keep, delete = sir.select_files_to_keep(similar)

        self.assertEqual(keep, [a])
        self.assertCountEqual(delete, [b, c])

    def test_tie_keeps_exactly_one(self):
        """Identical resolutions keep one file and delete the rest."""
        dupe_a = self._make_image("dupe_a.png", 1000, 1000)
        dupe_b = self._make_image("dupe_b.png", 1000, 1000)
        dupe_c = self._make_image("dupe_c.png", 1000, 1000)

        similar = {dupe_a: [(dupe_b, 1.0), (dupe_c, 1.0)]}
        keep, delete = sir.select_files_to_keep(similar, seed=7)

        self.assertEqual(len(keep), 1)
        self.assertCountEqual(keep + delete, [dupe_a, dupe_b, dupe_c])
        self.assertNotIn(keep[0], delete)

    def test_tie_is_reproducible_with_seed(self):
        """The same seed picks the same winner every run."""
        a = self._make_image("a.png", 900, 600)
        b = self._make_image("b.png", 900, 600)
        similar = {a: [(b, 1.0)]}

        first = sir.select_files_to_keep(similar, seed=1234)
        second = sir.select_files_to_keep(similar, seed=1234)

        self.assertEqual(first, second)

    def test_unreadable_file_loses_to_readable_one(self):
        """A corrupt or missing file never beats a valid image."""
        good = self._make_image("good.png", 300, 300)
        broken = os.path.join(self.test_dir, "broken.png")
        with open(broken, "w") as handle:
            handle.write("not an image")

        similar = {broken: [(good, 0.99)]}
        keep, delete = sir.select_files_to_keep(similar)

        self.assertEqual(keep, [good])
        self.assertEqual(delete, [broken])

    def test_files_without_matches_untouched(self):
        """Files absent from the similarity map are not reported at all."""
        a = self._make_image("a.png", 100, 100)
        b = self._make_image("b.png", 200, 200)
        lone = self._make_image("lone.png", 300, 300)

        similar = {a: [(b, 0.95)]}
        keep, delete = sir.select_files_to_keep(similar)

        self.assertEqual(keep, [b])
        self.assertEqual(delete, [a])
        self.assertNotIn(lone, keep + delete)


class TestNoCliStateDependency(unittest.TestCase):
    """Library functions must not depend on argparse state from main()."""

    def setUp(self):
        """Create one small image to operate on."""
        self.test_dir = tempfile.mkdtemp()
        self.image = os.path.join(self.test_dir, "a.png")
        Image.new("RGB", (64, 64), (1, 2, 3)).save(self.image)

    def tearDown(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_hash_failure_warns_without_cli_args(self):
        """A failing hash method must warn, not raise NameError, outside main()."""
        with mock.patch.object(
            sir, "calculate_single_hash", side_effect=RuntimeError("boom")
        ):
            hashes = sir.calculate_hashes(self.test_dir, verbose=False)

        self.assertEqual(hashes, {})

    def test_orb_error_without_cli_args(self):
        """An ORB failure must fall back, not raise NameError, outside main()."""
        with mock.patch("cv2.imread", side_effect=RuntimeError("boom")):
            result = sir.verify_with_orb(self.image, self.image, verbose=False)

        self.assertEqual(result, (True, 0, 1.0))


class TestIntegration(unittest.TestCase):
    """Integration tests using actual test_images directory."""

    @classmethod
    def setUpClass(cls):
        """Set up test images directory."""
        cls.test_images_dir = os.path.join(os.path.dirname(__file__), "test_images")

        # Create test images if they don't exist
        if not os.path.exists(cls.test_images_dir):
            try:
                import create_test_images

                print("\nCreating test images...")
                create_test_images.create_test_images(cls.test_images_dir)
            except ImportError:
                # The generator was removed in the repo cleanup; the individual
                # tests skip themselves when the directory is absent.
                print("\ntest_images/ absent and create_test_images.py not available")

    def test_integration_full_workflow(self):
        """Test the complete workflow."""
        if not os.path.exists(self.test_images_dir):
            self.skipTest("Test images directory not found")

        # Calculate hashes with multiple methods
        hashes = sir.calculate_hashes(
            self.test_images_dir, hash_methods=["ahash", "phash"]
        )
        self.assertGreater(len(hashes), 0)

        # Find similar images
        similar = sir.find_similar_images(self.test_images_dir, threshold=0.90)
        self.assertIsInstance(similar, dict)

        # Dry run delete
        deleted = sir.delete_similar_images(similar, dry_run=True)
        self.assertIsInstance(deleted, list)

    def test_integration_with_different_hash_methods(self):
        """Test with different hash methods."""
        if not os.path.exists(self.test_images_dir):
            self.skipTest("Test images directory not found")

        for method in ["ahash", "phash", "dhash", "whash"]:
            with self.subTest(method=method):
                similar = sir.find_similar_images(
                    self.test_images_dir, threshold=0.90, hash_methods=[method]
                )
                self.assertIsInstance(similar, dict)

    def test_integration_handles_various_formats(self):
        """Test handling of various image formats."""
        if not os.path.exists(self.test_images_dir):
            self.skipTest("Test images directory not found")

        hashes = sir.calculate_hashes(self.test_images_dir)

        # Should find at least the test files
        self.assertGreater(len(hashes), 0)

    def test_integration_subdirectories(self):
        """Test that subdirectories are handled correctly."""
        if not os.path.exists(self.test_images_dir):
            self.skipTest("Test images directory not found")

        subdir = os.path.join(self.test_images_dir, "subdir")
        if not os.path.exists(subdir):
            os.makedirs(subdir, exist_ok=True)
            img = Image.new("RGB", (100, 100), (255, 0, 0))
            img.save(os.path.join(subdir, "sub_image.png"))

        hashes = sir.calculate_hashes(self.test_images_dir)

        # Check that subdir images use full paths
        for path in hashes.keys():
            self.assertTrue(os.path.isabs(path))
            self.assertTrue(os.path.exists(path))


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def test_single_image(self):
        """Test with only one image."""
        test_dir = tempfile.mkdtemp()
        try:
            img = Image.new("RGB", (100, 100), (255, 0, 0))
            img.save(os.path.join(test_dir, "only.png"))

            similar = sir.find_similar_images(test_dir)
            self.assertEqual(len(similar), 0)
        finally:
            shutil.rmtree(test_dir)

    def test_corrupted_image(self):
        """Test handling of corrupted image."""
        test_dir = tempfile.mkdtemp()
        try:
            # Create a valid image
            img = Image.new("RGB", (100, 100), (255, 0, 0))
            img.save(os.path.join(test_dir, "valid.png"))

            # Create a corrupted file (pretending to be an image)
            with open(os.path.join(test_dir, "corrupted.png"), "w") as f:
                f.write("This is not an image")

            # Should not crash
            hashes = sir.calculate_hashes(test_dir)
            self.assertEqual(len(hashes), 1)  # Only the valid one
        finally:
            shutil.rmtree(test_dir)

    def test_permission_error_handling(self):
        """Test handling of permission errors during delete."""
        test_dir = tempfile.mkdtemp()
        try:
            img = Image.new("RGB", (100, 100), (255, 0, 0))
            img.save(os.path.join(test_dir, "file.png"))

            similar = {
                os.path.join(test_dir, "original.png"): [
                    (os.path.join(test_dir, "file.png"), 0.95)
                ]
            }

            # Should not crash even with permission issues
            deleted = sir.delete_similar_images(similar, dry_run=True)
            self.assertIsInstance(deleted, list)
        finally:
            shutil.rmtree(test_dir)

    def test_calculate_single_hash(self):
        """Test calculating single hash for an image."""
        img = Image.new("RGB", (100, 100), (255, 0, 0))

        for method in ["ahash", "phash", "dhash", "dhash_vertical", "whash"]:
            with self.subTest(method=method):
                hash_val = sir.calculate_single_hash(img, method)
                self.assertIsNotNone(hash_val)


class TestArgumentParsing(unittest.TestCase):
    """Tests for command-line argument parsing."""

    def test_main_with_valid_args(self):
        """Test main function with valid arguments."""
        test_dir = tempfile.mkdtemp()
        try:
            # Create at least one image
            img = Image.new("RGB", (100, 100), (255, 0, 0))
            img.save(os.path.join(test_dir, "test.png"))

            # Test with valid directory
            with mock.patch("sys.argv", ["script", "-d", test_dir]):
                # Should not raise
                try:
                    sir.main()
                except SystemExit as e:
                    # Expected for successful run
                    pass
        finally:
            shutil.rmtree(test_dir)

    def test_web_without_credentials_refuses_to_start(self):
        """--web without auth material exits 2 instead of serving anonymously."""
        with mock.patch("sys.argv", ["script", "--web"]):
            with mock.patch.dict(os.environ, {"SIR_PASSWORD": ""}):
                self.assertEqual(sir.main(), 2)


class TestHashSimilarityScaling(unittest.TestCase):
    """Similarity must be measured against the real hash width.

    The old divisor was a fixed 64 bits. At the CLI default hash_size=32 a hash
    is 1024 bits, so any pair whose distance exceeded 64 bits scored 0.0 - a
    near-duplicate looked completely unrelated. These pin the correct scale and
    the unchanged 8x8 behaviour.
    """

    def test_32x32_uses_1024_bit_scale(self):
        from imagehash import ImageHash
        import numpy as np

        first = ImageHash(np.zeros((32, 32), dtype=bool))
        second = ImageHash(np.zeros((32, 32), dtype=bool))
        second.hash[:8, :] = True  # 8 rows * 32 columns = 256 differing bits

        self.assertAlmostEqual(sir.hash_similarity(first, second), 0.75, places=6)

    def test_32x32_full_distance_is_zero(self):
        from imagehash import ImageHash
        import numpy as np

        first = ImageHash(np.zeros((32, 32), dtype=bool))
        second = ImageHash(np.ones((32, 32), dtype=bool))
        self.assertEqual(sir.hash_similarity(first, second), 0.0)

    def test_8x8_behaviour_is_unchanged(self):
        from imagehash import ImageHash
        import numpy as np

        first = ImageHash(np.zeros((8, 8), dtype=bool))
        second = ImageHash(np.zeros((8, 8), dtype=bool))
        second.hash[:4, :] = True  # half of 64 bits

        self.assertAlmostEqual(sir.hash_similarity(first, second), 0.5, places=6)

    def test_colorhash_uses_its_own_bit_count(self):
        from imagehash import ImageHash
        import numpy as np

        # colorhash(binbits=3) is 14 x 3 = 42 bits, not 64.
        first = ImageHash(np.zeros((14, 3), dtype=bool))
        second = ImageHash(np.zeros((14, 3), dtype=bool))
        second.hash[:7, :] = True  # 21 of 42 bits

        self.assertAlmostEqual(sir.hash_similarity(first, second), 0.5, places=6)

    def test_near_duplicate_survives_the_cli_hash_size(self):
        """The exact regression: 32x32 near-duplicates must not collapse to 0."""
        import numpy as np

        directory = tempfile.mkdtemp()
        try:
            rng = np.random.default_rng(3)
            array = rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)
            original = Image.fromarray(array)
            original.save(os.path.join(directory, "a.png"))
            original.resize((1200, 900)).save(
                os.path.join(directory, "b.jpg"), quality=88
            )

            hashes = sir.calculate_hashes(
                directory, hash_methods=["ahash", "phash"], hash_size=32
            )
            paths = sorted(hashes)
            similarity = sir.compute_combined_similarity(
                hashes[paths[0]], hashes[paths[1]]
            )
            self.assertGreater(similarity, 0.5)
        finally:
            shutil.rmtree(directory)


class TestProgressCallbacks(unittest.TestCase):
    """The web scan needs progress without tqdm; the CLI must keep tqdm."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        import numpy as np

        rng = np.random.default_rng(11)
        for index in range(3):
            Image.fromarray(
                rng.integers(0, 255, (120, 120, 3), dtype=np.uint8)
            ).save(os.path.join(self.directory, f"img{index}.png"))

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_calculate_hashes_reports_hashing_progress(self):
        events = []
        sir.calculate_hashes(
            self.directory,
            hash_methods=["ahash"],
            progress=lambda *args: events.append(args),
        )
        self.assertTrue(events)
        self.assertTrue(all(event[0] == "hashing" for event in events))
        self.assertEqual(events[-1][1], events[-1][2])
        self.assertEqual(events[-1][2], 3)

    def test_find_similar_images_reports_comparing_progress(self):
        events = []
        sir.find_similar_images(
            self.directory,
            threshold=0.0,
            hash_methods=["ahash"],
            progress=lambda *args: events.append(args),
        )
        stages = {event[0] for event in events}
        self.assertIn("hashing", stages)
        self.assertIn("comparing", stages)
        comparing = [event for event in events if event[0] == "comparing"]
        self.assertEqual(comparing[-1][1], comparing[-1][2])


class TestTieReport(unittest.TestCase):
    """select_files_to_keep can route its random-tie notice to a collector."""

    def test_report_callback_receives_tie_message(self):
        directory = tempfile.mkdtemp()
        try:
            first = os.path.join(directory, "a.png")
            second = os.path.join(directory, "b.png")
            Image.new("RGB", (100, 100), (5, 6, 7)).save(first)
            Image.new("RGB", (100, 100), (5, 6, 7)).save(second)

            messages = []
            keep, delete = sir.select_files_to_keep(
                {first: [(second, 1.0)]},
                seed=4,
                report=messages.append,
            )
            self.assertEqual(len(keep), 1)
            self.assertEqual(len(delete), 1)
            self.assertTrue(any("tied at 100x100" in message for message in messages))
        finally:
            shutil.rmtree(directory)


class TestWebHelpers(unittest.TestCase):
    """Pure-ish helpers behind the web UI."""

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        import numpy as np

        rng = np.random.default_rng(21)
        self.paths = {}
        for name, size in (("large.png", 240), ("mid.png", 160), ("small.png", 100)):
            path = self.directory / name
            Image.fromarray(
                rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            ).save(path)
            self.paths[name] = path

    def tearDown(self):
        shutil.rmtree(self.directory)

    def _state(self, pairs, protected=None):
        return {
            "dataset": str(self.directory.resolve()),
            "pairs": pairs,
            "scanned_at": time.time(),
            "config": {"seed_used": 7, "threshold": 0.9},
            "protected": protected or [],
            "resolutions": {"large.png": [240, 240], "mid.png": [160, 160],
                            "small.png": [100, 100]},
            "sizes": {"large.png": 1000, "mid.png": 600, "small.png": 300},
        }

    def test_build_results_keeps_largest_and_counts_reclaimable(self):
        from web import server as web

        state = self._state(
            [["small.png", "large.png", 0.95], ["mid.png", "large.png", 0.96]]
        )
        payload = web.build_results(self.directory, state, 0.90)

        self.assertEqual(payload["stats"]["groups"], 1)
        group = payload["groups"][0]
        self.assertEqual(group["keep"], "large.png")
        self.assertCountEqual(group["deletable"], ["mid.png", "small.png"])
        self.assertEqual(group["reclaimable"], 900)  # 600 + 300

    def test_build_results_excludes_protected_from_deletable(self):
        from web import server as web

        state = self._state(
            [["small.png", "large.png", 0.95], ["mid.png", "large.png", 0.96]],
            protected=["small.png"],
        )
        payload = web.build_results(self.directory, state, 0.90)
        group = payload["groups"][0]
        self.assertEqual(group["deletable"], ["mid.png"])
        self.assertEqual(group["reclaimable"], 600)

    def test_build_results_respects_threshold(self):
        from web import server as web

        state = self._state([["small.png", "large.png", 0.95]])
        self.assertEqual(web.build_results(self.directory, state, 0.99)["stats"]["groups"], 0)
        self.assertGreaterEqual(
            web.build_results(self.directory, state, 0.95)["stats"]["groups"], 1
        )

    def test_build_results_clamps_below_pair_floor(self):
        from web import server as web

        state = self._state([["small.png", "large.png", 0.60]])
        payload = web.build_results(self.directory, state, 0.20)
        self.assertTrue(payload["clamped"])
        self.assertEqual(payload["threshold"], web.PAIR_FLOOR)

    def test_validate_refuses_to_empty_a_group(self):
        from web import server as web

        state = self._state(
            [["small.png", "large.png", 0.95], ["mid.png", "large.png", 0.96]]
        )
        allowed, errors = web.validate_deletions(
            self.directory, state, ["large.png", "mid.png", "small.png"], 0.90
        )
        self.assertEqual(allowed, [])
        self.assertTrue(errors)

    def test_validate_refuses_protected_file(self):
        from web import server as web

        state = self._state([["small.png", "large.png", 0.95]], protected=["small.png"])
        allowed, errors = web.validate_deletions(
            self.directory, state, ["small.png"], 0.90
        )
        self.assertEqual(allowed, [])
        self.assertIn("pinned", errors["small.png"])

    def test_validate_allows_a_loser(self):
        from web import server as web

        state = self._state([["small.png", "large.png", 0.95]])
        allowed, errors = web.validate_deletions(
            self.directory, state, ["small.png"], 0.90
        )
        self.assertEqual(allowed, ["small.png"])
        self.assertEqual(errors, {})

    def test_validate_rejects_path_traversal(self):
        from web import server as web

        state = self._state([["small.png", "large.png", 0.95]])
        allowed, errors = web.validate_deletions(
            self.directory, state, ["../outside.png"], 0.90
        )
        self.assertEqual(allowed, [])
        self.assertTrue(errors)

    def test_prune_missing_drops_edges_and_pins(self):
        from web import server as web

        state = self._state(
            [["small.png", "large.png", 0.95]], protected=["small.png"]
        )
        self.paths["small.png"].unlink()
        missing = web.prune_missing(self.directory, state)
        self.assertEqual(missing, ["small.png"])
        self.assertEqual(state["pairs"], [])
        self.assertEqual(state["protected"], [])
        self.assertNotIn("small.png", state["sizes"])

    def test_normalise_config_rejects_unknown_method(self):
        from web import server as web

        config, error = web.normalise_config(
            {"hash_methods": ["bogus"]}, web.DEFAULT_CONFIG
        )
        self.assertIsNotNone(error)
        self.assertIn("unknown hash method", error)

    def test_normalise_config_accepts_valid_settings(self):
        from web import server as web

        config, error = web.normalise_config(
            {
                "threshold": 0.83,
                "hash_methods": ["dhash", "whash"],
                "hash_size": 16,
                "verify_orb": True,
                "orb_min_matches": 20,
                "orb_threshold": 0.7,
                "seed": 123,
            },
            web.DEFAULT_CONFIG,
        )
        self.assertIsNone(error)
        self.assertEqual(config["threshold"], 0.83)
        self.assertEqual(config["hash_methods"], ["dhash", "whash"])
        self.assertEqual(config["seed"], 123)
        self.assertEqual(config["pair_floor"], web.PAIR_FLOOR)

    def test_normalise_config_lowers_pair_floor_for_low_thresholds(self):
        """A scan at 30% must cache down to 30%, or the UI can never show it."""
        from web import server as web

        config, error = web.normalise_config(
            {"threshold": 0.30}, web.DEFAULT_CONFIG
        )
        self.assertIsNone(error)
        self.assertAlmostEqual(config["pair_floor"], 0.30)

    def test_build_results_keeps_the_same_winner_across_calls(self):
        """A blank seed is fixed at scan time; reloading must not re-roll it."""
        from web import server as web

        tie = self.directory / "tie.png"
        Image.new("RGB", (240, 240), (9, 9, 9)).save(tie)
        state = self._state(
            [["large.png", "tie.png", 1.0], ["large.png", "small.png", 0.95]]
        )
        state["resolutions"]["tie.png"] = [240, 240]
        state["sizes"]["tie.png"] = 1000

        first = web.build_results(self.directory, state, 0.90)
        second = web.build_results(self.directory, state, 0.90)
        self.assertEqual(first["groups"][0]["keep"], second["groups"][0]["keep"])
        self.assertTrue(first["groups"][0]["tie"])

    def test_build_results_uses_the_states_own_floor(self):
        from web import server as web

        state = self._state([["small.png", "large.png", 0.35]])
        state["config"]["pair_floor"] = 0.35
        low = web.build_results(self.directory, state, 0.35)
        self.assertEqual(low["stats"]["groups"], 1)
        lower = web.build_results(self.directory, state, 0.20)
        self.assertTrue(lower["clamped"])
        self.assertAlmostEqual(lower["threshold"], 0.35)


class TestWebServer(unittest.TestCase):
    """End-to-end HTTP tests against a real server on a free port."""

    def setUp(self):
        import numpy as np
        from web import server as web

        self.web = web
        self.tmp = tempfile.mkdtemp()
        self.dataset = Path(self.tmp) / "dataset"
        self.dataset.mkdir()
        rng = np.random.default_rng(33)
        array = rng.integers(0, 255, (900, 1200, 3), dtype=np.uint8)
        image = Image.fromarray(array)
        image.save(self.dataset / "large.png")
        image.resize((600, 450)).save(self.dataset / "small.png", quality=85)
        Image.fromarray(
            rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
        ).save(self.dataset / "other.png")

        self.old_runs = web.RUNS_ROOT
        web.RUNS_ROOT = Path(self.tmp) / "runs"
        web.RUNS_ROOT.mkdir()
        web.STATES.clear()
        web.cancel_scan(mark_cancelled=False)
        web.set_job(phase="idle", stage="", done=0, total=0, current="",
                    error=None, started=None, finished=None)
        with web.CFG_LOCK:
            web.CFG.clear()
            web.CFG.update(
                {
                    "dataset": None,
                    "slug": None,
                    "auth": False,
                    "username": "",
                    "password": "",
                    "defaults": dict(web.DEFAULT_CONFIG),
                    "scan_proc": None,
                    "scan_config": None,
                }
            )

        self.server, _, _ = web.choose_server("127.0.0.1", 0)
        self.assertIsNotNone(self.server)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.05},
            daemon=True,
        )
        self.thread.start()

    def tearDown(self):
        web = self.web
        web.cancel_scan(mark_cancelled=False)
        self.server.shutdown()
        self.server.server_close()
        web.RUNS_ROOT = self.old_runs
        web.STATES.clear()
        shutil.rmtree(self.tmp)

    def _request(self, method, path, body=None, headers=None, auth=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data, method=method
        )
        if body is not None:
            request.add_header("Content-Type", "application/json")
        request.add_header("X-Requested-With", "similar-image-remover")
        if auth is not None:
            token = base64.b64encode(auth.encode()).decode()
            request.add_header("Authorization", "Basic " + token)
        for key, value in (headers or {}).items():
            request.add_header(key, value)
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = response.read()
                return response.status, json.loads(payload or b"{}"), dict(response.headers)
        except urllib.error.HTTPError as error:
            payload = error.read()
            try:
                parsed = json.loads(payload or b"{}")
            except json.JSONDecodeError:
                parsed = payload.decode("utf-8", "replace")
            return error.code, parsed, dict(error.headers)

    def _wait_for_scan(self):
        deadline = time.time() + 20
        while time.time() < deadline:
            status, state, _ = self._request("GET", "/api/state")
            self.assertEqual(status, 200)
            phase = state["job"]["phase"]
            if phase not in ("scanning", "loading"):
                return state
            time.sleep(0.05)
        self.fail("scan did not finish")

    def test_ping_and_state_round_trip(self):
        status, payload, _ = self._request("GET", "/api/ping")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"ok": True, "dataset": None, "auth": False})

        status, state, _ = self._request("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertIsNone(state["dataset"])

    def test_dataset_scan_results_and_individual_delete(self):
        status, state, _ = self._request(
            "POST", "/api/dataset", {"path": str(self.dataset)}
        )
        self.assertEqual(status, 200)
        self.assertEqual(state["dataset"]["path"], str(self.dataset.resolve()))

        state = self._wait_for_scan()
        self.assertEqual(state["job"]["phase"], "done")
        self.assertTrue(state["has_results"])

        status, results, _ = self._request("GET", "/api/results?threshold=0.9")
        self.assertEqual(status, 200)
        self.assertEqual(results["stats"]["groups"], 1)
        group = results["groups"][0]
        self.assertEqual(group["keep"], "large.png")
        self.assertIn("small.png", group["deletable"])

        status, reply, _ = self._request(
            "POST", "/api/delete", {"paths": ["small.png"], "threshold": 0.9}
        )
        self.assertEqual(status, 200)
        self.assertEqual(reply["deleted"], ["small.png"])
        self.assertFalse((self.dataset / "small.png").exists())

        status, results, _ = self._request("GET", "/api/results?threshold=0.9")
        self.assertEqual(status, 200)
        self.assertEqual(results["stats"]["groups"], 0)

    def test_protect_blocks_bulk_delete(self):
        self._request("POST", "/api/dataset", {"path": str(self.dataset)})
        self._wait_for_scan()

        status, reply, _ = self._request(
            "POST", "/api/protect", {"path": "small.png", "protected": True}
        )
        self.assertEqual(status, 200)
        self.assertEqual(reply["protected"], ["small.png"])

        status, reply, _ = self._request(
            "POST", "/api/delete",
            {"paths": ["small.png", "large.png"], "threshold": 0.9},
        )
        self.assertEqual(status, 400)
        self.assertIn("details", reply)
        self.assertTrue((self.dataset / "small.png").exists())
        self.assertTrue((self.dataset / "large.png").exists())

    def test_basic_auth_challenge_and_credentials(self):
        web = self.web
        with web.CFG_LOCK:
            web.CFG["auth"] = True
            web.CFG["username"] = "demo"
            web.CFG["password"] = "secret"

        status, payload, headers = self._request("GET", "/api/ping")
        self.assertEqual(status, 401)
        self.assertIn("Basic", headers.get("WWW-Authenticate", ""))
        self.assertEqual(payload, "Authentication required\n")

        status, _, _ = self._request("GET", "/api/ping", auth="demo:wrong")
        self.assertEqual(status, 401)

        status, payload, _ = self._request("GET", "/api/ping", auth="demo:secret")
        self.assertEqual(status, 200)
        self.assertTrue(payload["auth"])

    def test_cross_origin_post_is_refused(self):
        data = json.dumps({"path": str(self.dataset)}).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/dataset", data=data, method="POST"
        )
        request.add_header("Content-Type", "application/json")
        # Deliberately no X-Requested-With header.
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=10)
        self.assertEqual(caught.exception.code, 403)

    def test_image_path_traversal_is_refused(self):
        self._request("POST", "/api/dataset", {"path": str(self.dataset)})
        self._wait_for_scan()
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request("GET", "/image/../../../../etc/passwd")
            response = connection.getresponse()
            self.assertIn(response.status, (400, 404))
        finally:
            connection.close()

    def test_thumb_and_preview_are_smaller_than_the_original(self):
        self._request("POST", "/api/dataset", {"path": str(self.dataset)})
        self._wait_for_scan()

        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}/thumb/large.png", timeout=10
        ) as response:
            thumb_bytes = response.read()
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}/preview/large.png", timeout=10
        ) as response:
            preview_bytes = response.read()
        original_bytes = (self.dataset / "large.png").read_bytes()

        self.assertGreater(len(thumb_bytes), 0)
        self.assertGreater(len(preview_bytes), len(thumb_bytes))
        self.assertLess(len(thumb_bytes), len(original_bytes))


if __name__ == "__main__":
    # First ensure test images exist
    test_images_dir = os.path.join(os.path.dirname(__file__), "test_images")
    if not os.path.exists(test_images_dir):
        print("Creating test images...")
        try:
            import create_test_images

            create_test_images.create_test_images(test_images_dir)
        except ImportError:
            # The generator was removed in commit 862e056 and test_images/ is
            # gitignored, so this cannot succeed. Continue anyway: the
            # integration tests skip themselves and the rest still run, which
            # beats dying with a traceback before a single test executes.
            print(
                "create_test_images.py is gone; integration tests will be skipped."
            )

    # Run tests
    unittest.main(verbosity=2)
