# Hazardous‑Material Sign & Barrel Detection in a Robotic Environment

This project uses **OpenCV** with the **SIFT** algorithm to detect hazardous‑material symbols (hazmat signs) and coloured barrels (red / blue) in a video file.

---

## Requirements

- Python 3.x  
- OpenCV → `opencv-contrib-python`  
- NumPy  

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Folder Structure

```
main.py
video.mp4
detections/        # Frames with detections will be saved here
hazmats/           # Template PNG files (hazmat symbols)
```

---

## Usage

1. Place PNG templates of the hazmat signs you want to detect inside the `hazmats/` folder.  
2. Put the video you want to analyse in the project root as `video.mp4`.  
3. Run the script:

   ```bash
   python main.py
   ```

4. When prompted, enter the frame interval (e.g. **20**) or press *Enter* to accept the default (recommended) value.  
5. Detected objects are shown on‑screen and saved to the `detections/` folder.  
   Press any key to continue after each displayed detection.  
   *(If the folder does not exist, nothing is saved.)*

---

## Notes

- A screenshot is taken for every detected object and written to `detections/`.  
- Red and blue barrels are detected via colour‑based segmentation.  
- Hazmat symbols are detected using **SIFT** keypoint matching and template matching.
