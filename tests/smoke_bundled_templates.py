from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import cv2
import numpy as np


class BundledTemplateSmoke(unittest.TestCase):
    def test_default_templates_work_from_another_directory(self):
        script_path = Path(__file__).resolve().parents[1] / "main.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "fixture.avi"
            output_dir = root / "new" / "detections"
            red = np.zeros((128, 128, 3), np.uint8)
            red[20:100, 20:100] = (0, 0, 255)
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (128, 128)
            )
            self.assertTrue(writer.isOpened(), "The MJPEG fixture encoder must be available")
            try:
                writer.write(red)
            finally:
                writer.release()
            started = time.monotonic()
            result = subprocess.run(
                [sys.executable, str(script_path), "--headless", "--frame-interval", "1",
                 "--video", str(video_path), "--output", str(output_dir)],
                cwd=root, capture_output=True, text=True, check=False, timeout=120
            )
            elapsed = time.monotonic() - started
            self.assertEqual(result.returncode, 0, result.stderr)
            snapshot = output_dir / "varil_Kirmizi Varil_0.png"
            self.assertTrue(snapshot.is_file(), result.stdout)
            decoded = cv2.imread(str(snapshot))
            self.assertIsNotNone(decoded)
            self.assertEqual(decoded.shape, red.shape)
            self.assertIn("Detected: Kirmizi Varil", result.stdout)
            print(f"Bundled-template CLI smoke completed in {elapsed:.3f} seconds.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
