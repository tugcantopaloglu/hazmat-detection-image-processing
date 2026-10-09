import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import main


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.templates_dir = self.root / "templates"
        self.templates_dir.mkdir()
        self.template = np.random.default_rng(7).integers(
            0, 256, size=(128, 128), dtype=np.uint8
        )
        self.assertTrue(cv2.imwrite(str(self.templates_dir / "fixture.png"), self.template))
        self.output_dir = self.root / "new" / "detections"
        self.cli_environment = dict(os.environ, OMP_NUM_THREADS="1",
                                    OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def write_video(self, frames):
        video_path = self.root / "fixture.avi"
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (width, height)
        )
        self.assertTrue(writer.isOpened(), "The MJPEG fixture encoder must be available")
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        return video_path

    def test_missing_and_empty_template_folders_fail(self):
        with self.assertRaises(ValueError):
            main.load_templates(self.root / "missing")
        empty_dir = self.root / "empty"
        empty_dir.mkdir()
        with self.assertRaises(ValueError):
            main.load_templates(empty_dir)

    def test_unreadable_and_featureless_templates_fail(self):
        invalid_dir = self.root / "invalid"
        invalid_dir.mkdir()
        (invalid_dir / "invalid.png").write_text("invalid image", encoding="utf-8")
        self.assertTrue(cv2.imwrite(str(invalid_dir / "blank.png"), np.zeros((32, 32), np.uint8)))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            main.load_templates(invalid_dir)

    def test_single_neighbor_does_not_crash_matching(self):
        class FixedSift:
            def detectAndCompute(self, image, mask):
                return [], np.ones((2, 128), np.float32)

        class SingleNeighborMatcher:
            def knnMatch(self, query, train, k):
                return [[cv2.DMatch(0, 0, 0.1)]]

        templates = {"fixture": {"des": np.ones((1, 128), np.float32)}}
        self.assertEqual(
            main.detect_templates(self.template, templates, FixedSift(), SingleNeighborMatcher()),
            [],
        )

    def test_bundled_template_matches_and_blank_frame_does_not(self):
        bundled_dir = Path(main.__file__).resolve().parent / "hazmats"
        image = cv2.imread(str(bundled_dir / "corrosive.png"), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image)
        image = cv2.resize(image, (256, 256))
        self.assertTrue(cv2.imwrite(str(self.templates_dir / "corrosive.png"), image))
        templates, sift = main.load_templates(self.templates_dir)
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        detections = main.detect_templates(image, templates, sift, matcher)
        self.assertIn("corrosive", [label for label, box in detections])
        blank = np.zeros_like(image)
        self.assertEqual(main.detect_templates(blank, templates, sift, matcher), [])

    def test_fewer_than_four_matches_skip_homography(self):
        class FixedSift:
            def detectAndCompute(self, image, mask):
                keypoints = [cv2.KeyPoint(float(index), float(index), 1) for index in range(3)]
                return keypoints, np.ones((3, 128), np.float32)

        class ThreeMatchMatcher:
            def knnMatch(self, query, train, k):
                return [[cv2.DMatch(index, index, 0.1), cv2.DMatch(index, index, 1)]
                        for index in range(3)]

        keypoints, descriptors = FixedSift().detectAndCompute(self.template, None)
        templates = {"fixture": {"des": descriptors, "kp": keypoints, "img": self.template}}
        self.assertEqual(main.detect_templates(self.template, templates, FixedSift(),
                                              ThreeMatchMatcher(), match_threshold=0), [])

    def test_color_regions_are_detected(self):
        frame = np.zeros((128, 128, 3), np.uint8)
        frame[10:40, 10:40] = (0, 0, 255)
        frame[70:110, 70:110] = (255, 0, 0)
        self.assertEqual(
            main.detect_color_objects(frame),
            [("Kirmizi Varil", (10, 10, 30, 30)), ("Mavi Varil", (70, 70, 40, 40))],
        )

    def test_invalid_frame_intervals_fail_before_opening_video(self):
        for interval in (0, -1):
            with self.subTest(interval=interval), patch.object(cv2, "VideoCapture") as capture:
                with self.assertRaises(ValueError):
                    main.main("missing.avi", self.templates_dir, frame_interval=interval, display=False)
                capture.assert_not_called()

    def test_headless_pipeline_creates_snapshots_and_resets_flags(self):
        red = np.zeros((128, 128, 3), np.uint8)
        red[20:100, 20:100] = (0, 0, 255)
        blank = np.zeros_like(red)
        video_path = self.write_video([red, red, blank, red])
        with patch.object(cv2, "imshow", side_effect=AssertionError("Unexpected GUI call")), \
             patch.object(cv2, "waitKey", side_effect=AssertionError("Unexpected GUI call")), \
             contextlib.redirect_stdout(io.StringIO()):
            main.main(video_path, self.templates_dir, frame_interval=1,
                      output_dir=self.output_dir, display=False)
        snapshots = sorted(self.output_dir.glob("varil_Kirmizi Varil_*.png"))
        self.assertEqual([path.name for path in snapshots],
                         ["varil_Kirmizi Varil_0.png", "varil_Kirmizi Varil_3.png"])
        for snapshot in snapshots:
            self.assertEqual(cv2.imread(str(snapshot)).shape, red.shape)

    def test_annotations_do_not_change_color_detection_input(self):
        red = np.zeros((128, 128, 3), np.uint8)
        red[20:100, 20:100] = (0, 0, 255)
        video_path = self.write_video([red])
        polygon = np.float32([[55, 10], [55, 110], [65, 110], [65, 10]]).reshape(-1, 1, 2)
        color_inputs = []
        original_detector = main.detect_color_objects

        def record_detections(frame):
            color_inputs.append(original_detector(frame))
            return color_inputs[-1]

        with patch.object(main, "detect_templates", return_value=[("fixture", polygon)]), \
             patch.object(main, "detect_color_objects", side_effect=record_detections), \
             contextlib.redirect_stdout(io.StringIO()):
            main.main(video_path, self.templates_dir, frame_interval=1,
                      output_dir=self.output_dir, display=False)
        self.assertEqual(len(color_inputs[0]), 1)
        label, (x, y, width, height) = color_inputs[0][0]
        self.assertEqual(label, "Kirmizi Varil")
        self.assertGreater(width, 75)

    def test_failed_snapshot_write_releases_capture(self):
        red = np.zeros((128, 128, 3), np.uint8)
        red[20:100, 20:100] = (0, 0, 255)
        video_path = self.write_video([red])
        capture = cv2.VideoCapture(str(video_path))
        with patch.object(cv2, "VideoCapture", return_value=capture), \
             patch.object(cv2, "imwrite", return_value=False), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(OSError):
                main.main(video_path, self.templates_dir, frame_interval=1,
                          output_dir=self.output_dir, display=False)
        self.assertFalse(capture.isOpened())

    def test_cli_supports_other_working_directories_and_reports_input_errors(self):
        script_path = str(Path(main.__file__).resolve())
        help_result = subprocess.run(
            [sys.executable, script_path, "--help"], cwd=self.root,
            capture_output=True, text=True, check=False, timeout=20, env=self.cli_environment
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        invalid_interval = subprocess.run(
            [sys.executable, script_path, "--headless", "--frame-interval", "0"], cwd=self.root,
            capture_output=True, text=True, check=False, timeout=20, env=self.cli_environment
        )
        self.assertEqual(invalid_interval.returncode, 2)
        self.assertIn("positive integer", invalid_interval.stderr)
        missing_video = subprocess.run(
            [sys.executable, script_path, "--headless", "--video", str(self.root / "missing.avi"),
             "--templates", str(self.templates_dir)], cwd=self.root,
            capture_output=True, text=True, check=False, timeout=20, env=self.cli_environment
        )
        self.assertEqual(missing_video.returncode, 1)
        self.assertIn("could not be opened", missing_video.stderr)
        self.assertNotIn("Traceback", missing_video.stderr)
        red = np.zeros((128, 128, 3), np.uint8)
        red[20:100, 20:100] = (0, 0, 255)
        video_path = self.write_video([red])
        valid_video = subprocess.run(
            [sys.executable, script_path, "--headless", "--frame-interval", "1",
             "--video", str(video_path), "--templates", str(self.templates_dir),
             "--output", str(self.output_dir)], cwd=self.root,
            capture_output=True, text=True, check=False, timeout=20, env=self.cli_environment
        )
        self.assertEqual(valid_video.returncode, 0, valid_video.stderr)
        self.assertTrue((self.output_dir / "varil_Kirmizi Varil_0.png").is_file())


if __name__ == "__main__":
    unittest.main()
