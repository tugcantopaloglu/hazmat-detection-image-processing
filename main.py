import argparse
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent


def load_templates(template_folder):
    template_folder = Path(template_folder)
    if not template_folder.is_dir():
        raise ValueError(f"Template folder does not exist: {template_folder}")
    templates = {}
    sift = cv2.SIFT_create()
    template_files = sorted(path for path in template_folder.iterdir()
                            if path.is_file() and path.suffix.lower() == ".png")
    for file_path in template_files:
        label = file_path.stem
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Warning: Template image could not be read:", file_path)
            continue
        kp, des = sift.detectAndCompute(img, None)
        if des is None:
            print("Warning: Template has no SIFT features:", file_path)
            continue
        templates[label] = {"img": img, "kp": kp, "des": des}
    if not templates:
        raise ValueError(f"No readable PNG templates with SIFT features found in: {template_folder}")
    return templates, sift

def detect_templates(frame_gray, templates, sift, flann, 
                     match_threshold=10, ratio_tolerance=0.9,
                     min_box_area=3000, min_inlier_ratio=0.8):
    detection_results = []
    frame_kp, frame_des = sift.detectAndCompute(frame_gray, None)
    if frame_des is None or len(frame_des) < 2:
        return detection_results

    for label, data in templates.items():
        if data["des"] is None:
            continue
        try:
            matches = flann.knnMatch(data["des"], frame_des, k=2)
        except cv2.error:
            continue
        good_matches = []
        for match_pair in matches:
            if len(match_pair) != 2:
                continue
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        if len(good_matches) > match_threshold and len(good_matches) >= 4:
            src_pts = np.float32([data["kp"][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None and mask is not None:
                inliers = np.sum(mask)
                total = len(mask)
                inlier_ratio = float(inliers) / total
                if inlier_ratio < min_inlier_ratio:
                    continue

                h, w = data["img"].shape
                pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts, M)
                if not np.isfinite(dst).all():
                    continue

                transformed = np.int32(dst)
                epsilon = 0.02 * cv2.arcLength(transformed, True)
                approx = cv2.approxPolyDP(transformed, epsilon, True)
                if len(approx) != 4:
                    continue
                if not cv2.isContourConvex(approx):
                    continue
                x, y, w_box, h_box = cv2.boundingRect(np.int32(dst))
                if h_box == 0:
                    continue
                box_ratio = float(w_box) / h_box
                template_ratio = float(data["img"].shape[1]) / data["img"].shape[0]
                if abs(box_ratio - template_ratio) > ratio_tolerance:
                    continue
                box_area = w_box * h_box
                if box_area < min_box_area:
                    continue
                detection_results.append((label, dst))
    return detection_results

def detect_color_objects(frame):
    detections = []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask_red1, mask_red2)
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append(("Kirmizi Varil", (x, y, w, h)))
    contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_blue:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append(("Mavi Varil", (x, y, w, h)))
    return detections

def main(video_path, template_folder, frame_interval=None,
         output_dir=PROJECT_DIR / "detections", display=True):
    if frame_interval is None:
        if display:
            try:
                user_input = input("Frame interval (positive integer, default 20): ").strip()
                frame_interval = int(user_input) if user_input else 20
            except (ValueError, EOFError) as error:
                raise ValueError("Provide a positive integer with --frame-interval.") from error
        else:
            frame_interval = 20
    if not isinstance(frame_interval, int) or frame_interval <= 0:
        raise ValueError("Frame interval must be a positive integer.")
    templates, sift = load_templates(template_folder)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    cap = cv2.VideoCapture(str(video_path))
    detection_flags = {}
    for label in templates.keys():
        detection_flags[label] = False
    detection_flags["Kirmizi Varil"] = False
    detection_flags["Mavi Varil"] = False
    frame_count = 0
    try:
        if not cap.isOpened():
            raise ValueError(f"Video file could not be opened: {video_path}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        while True:
            ret, frame = cap.read()
            if not ret:
                if frame_count == 0:
                    raise ValueError(f"Video contains no readable frames: {video_path}")
                break
            if frame_count % frame_interval == 0:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                template_detections = detect_templates(frame_gray, templates, sift, flann,
                                                       match_threshold=10, ratio_tolerance=0.7,
                                                       min_box_area=3000, min_inlier_ratio=0.8)
                color_detections = detect_color_objects(frame)
                for label, box in template_detections:
                    box_int = np.int32(box)
                    cv2.polylines(frame, [box_int], True, (0, 255, 0), 3, cv2.LINE_AA)
                    if not detection_flags.get(label, False):
                        save_detection(frame, output_dir / f"hazard_{label}_{frame_count}.png",
                                       label, display)
                        detection_flags[label] = True
                detected_labels = [det[0] for det in template_detections]
                for label in templates.keys():
                    if label not in detected_labels:
                        detection_flags[label] = False
                for label, (x, y, w, h) in color_detections:
                    rect_color = (255, 0, 0) if label == "Mavi Varil" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), rect_color, 2)
                    if not detection_flags.get(label, False):
                        save_detection(frame, output_dir / f"varil_{label}_{frame_count}.png",
                                       label, display)
                        detection_flags[label] = True
                for label in ["Kirmizi Varil", "Mavi Varil"]:
                    if not any(det[0] == label for det in color_detections):
                        detection_flags[label] = False
            if display:
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    break
            frame_count += 1
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()


def save_detection(frame, filename, label, display):
    if not cv2.imwrite(str(filename), frame):
        raise OSError(f"Detection image could not be saved: {filename}")
    print("Detected: " + label)
    if display:
        cv2.imshow("Frame", frame)
        print("Press any key to continue...")
        cv2.waitKey(0)


def positive_int(value):
    try:
        interval = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Frame interval must be a positive integer.") from error
    if interval <= 0:
        raise argparse.ArgumentTypeError("Frame interval must be a positive integer.")
    return interval


def cli():
    parser = argparse.ArgumentParser(description="Detect hazmat templates and red/blue regions in a video.")
    parser.add_argument("--video", type=Path, default=PROJECT_DIR / "video.mp4",
                        help="Input video (default: bundled video.mp4).")
    parser.add_argument("--templates", type=Path, default=PROJECT_DIR / "hazmats",
                        help="PNG template folder (default: bundled hazmats).")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "detections",
                        help="Detection image folder (created automatically).")
    parser.add_argument("--frame-interval", type=positive_int,
                        help="Process every Nth frame; default 20 in headless mode, otherwise prompt.")
    parser.add_argument("--headless", action="store_true", help="Process without windows or keypresses.")
    args = parser.parse_args()
    try:
        main(args.video, args.templates, frame_interval=args.frame_interval,
             output_dir=args.output, display=not args.headless)
    except (ValueError, OSError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(cli())
