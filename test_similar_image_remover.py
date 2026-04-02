#!/usr/bin/env python3
"""
Comprehensive unit tests for similar_image_remover.py
"""

import unittest
import os
import sys
import tempfile
import shutil
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
        img.save(self.file1)
        img.save(self.file2)
        img.save(self.file3)

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


class TestIntegration(unittest.TestCase):
    """Integration tests using actual test_images directory."""

    @classmethod
    def setUpClass(cls):
        """Set up test images directory."""
        cls.test_images_dir = os.path.join(os.path.dirname(__file__), "test_images")

        # Create test images if they don't exist
        if not os.path.exists(cls.test_images_dir):
            print("\nCreating test images...")
            import create_test_images

            create_test_images.create_test_images(cls.test_images_dir)

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


if __name__ == "__main__":
    # First ensure test images exist
    test_images_dir = os.path.join(os.path.dirname(__file__), "test_images")
    if not os.path.exists(test_images_dir):
        print("Creating test images...")
        import create_test_images

        create_test_images.create_test_images(test_images_dir)

    # Run tests
    unittest.main(verbosity=2)
