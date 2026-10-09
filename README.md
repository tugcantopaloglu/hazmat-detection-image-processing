# Hazardous Material Sign and Color Detection

This OpenCV prototype detects the supplied hazmat sign templates in a video using SIFT features, FLANN matching, and RANSAC homography estimation. It also finds red and blue regions using HSV thresholds and contour area filtering. It uses no learned model or downloaded weights.

Color regions are labeled as barrels by the original application, but the algorithm does not verify barrel shape or hazardous contents. The repository includes 15 PNG templates and a sample `video.mp4`. These assets are retained unchanged; they are not a labeled evaluation dataset, and no detection accuracy is established.

## Installation

Use a Python environment compatible with the selected OpenCV and NumPy wheels. From the repository folder:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on Linux/macOS, then install:

```sh
python -m pip install -r requirements.txt
```

The requirements select one OpenCV package, `opencv-contrib-python`, which includes the main modules. Do not install another OpenCV wheel in the same environment: these packages share the `cv2` namespace. See the [OpenCV package installation guidance](https://pypi.org/project/opencv-contrib-python/#installation-and-usage).

## Usage

```sh
python main.py
```

The script prompts for a positive frame interval; press Enter for 20. It processes frame 0 and every Nth frame after that, shows the video in an OpenCV window, and pauses on each newly detected label. Press any key to continue after a detection, or Escape during normal playback to stop. Desktop mode requires an available graphical display.

Bundled video and template paths, and the default `detections/` output folder, are resolved relative to `main.py`, so launching the script from another folder works. Explicit relative paths are resolved against your current working directory.

```sh
python main.py --video ./input.avi --templates ./hazmats --output ./results --frame-interval 10
```

For processing without windows, prompts, or keypresses:

```sh
python main.py --video ./input.avi --frame-interval 20 --headless
```

Headless mode uses interval 20 when `--frame-interval` is omitted. `python main.py --help` lists all options. A missing video, an unusable template folder, a nonpositive interval, or a failed snapshot write produces an error and a nonzero exit status.

The output folder is created automatically. Annotated PNG snapshots use `hazard_<label>_<frame>.png` for signs and `varil_<label>_<frame>.png` for color regions. The existing color labels are `Kirmizi Varil` (red) and `Mavi Varil` (blue). A snapshot is saved when a label first appears or reappears after being absent from a processed frame. Repeated objects with the same label in one frame share that label's detection flag. Running the same input again into the same folder can overwrite matching snapshot filenames.

## Detection limits and validation

SIFT matching depends on visible template features, scale, lighting, perspective, and image quality. Fixed color thresholds can flag unrelated red or blue objects. Frame sampling can miss brief appearances. Both detectors examine the original frame before annotations are drawn. This prototype does not establish suitability for robot navigation or safety decisions.

Run the regression tests from the repository folder after installing requirements:

```sh
python -m unittest discover -s tests -v
```

Tests use small generated images and an MJPEG video, plus a resized copy of one bundled sign. They cover invalid inputs, sparse matches, snapshot creation and write failures, detection flag resets, unmodified color inputs, and headless command-line operation. The CLI regression uses a small generated template to keep this suite fast. Fixture subprocesses set OpenMP/BLAS thread limits to 1 within their own environments.

Run the separate bundled-template smoke check to exercise the default template path from another working directory:

```sh
python tests/smoke_bundled_templates.py
```

This check loads all 15 bundled templates through the actual CLI and verifies a snapshot from a generated video. It has a 120-second subprocess timeout; slow template loading or a hang fails the check. These tests do not establish real-world accuracy or exercise desktop interaction or robotic hardware.

The existing [MIT license](LICENSE) and author attribution are retained.
