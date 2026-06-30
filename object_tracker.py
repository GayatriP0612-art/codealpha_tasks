#!/usr/bin/env python3
"""
Object Detection and Tracking
-------------------------------
Real-time object detection (YOLOv8, pre-trained) + multi-object tracking
(SORT algorithm) on a webcam stream or video file, using OpenCV for I/O
and display.

Pipeline:
1. Real-time video input  -> OpenCV captures from webcam or a video file
2. Pre-trained detector   -> YOLOv8 (ultralytics) detects objects per frame
3. Process each frame     -> bounding boxes + class labels extracted
4. Object tracking        -> SORT (Kalman filter + Hungarian assignment)
                              assigns a persistent ID to each tracked object
5. Display output         -> bounding boxes, class labels, and tracking IDs
                              drawn on the video in real time

Setup (run once in a terminal):
    pip install opencv-python ultralytics numpy scipy filterpy

Run on your webcam:
    python object_tracker.py --source 0

Run on a video file:
    python object_tracker.py --source path/to/video.mp4 --output tracked.mp4

Notes:
- The first run will auto-download the YOLOv8n weights (yolov8n.pt, ~6MB).
- Press 'q' to quit the live window.
"""

import argparse

import numpy as np
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("Missing dependency 'ultralytics'.\nInstall it with:  pip install ultralytics")

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    raise SystemExit("Missing dependency 'scipy'.\nInstall it with:  pip install scipy")

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    raise SystemExit("Missing dependency 'filterpy'.\nInstall it with:  pip install filterpy")


# ---------------------------------------------------------------------------
# SORT: Simple Online and Realtime Tracking
# (Kalman filter per track + IoU-based Hungarian assignment)
# ---------------------------------------------------------------------------
def iou(bb_test, bb_gt):
    """Intersection-over-union of two [x1,y1,x2,y2] boxes."""
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def bbox_to_z(bbox):
    """[x1,y1,x2,y2] -> [cx,cy,scale(area),aspect_ratio]."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h) if h != 0 else 0
    return np.array([cx, cy, s, r]).reshape((4, 1))


def x_to_bbox(x):
    """[cx,cy,scale,aspect_ratio] -> [x1,y1,x2,y2]."""
    w = np.sqrt(max(x[2] * x[3], 0))
    h = x[2] / w if w != 0 else 0
    return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))


class Track:
    """A single tracked object, backed by a Kalman filter."""

    count = 0

    def __init__(self, bbox, cls_name):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = bbox_to_z(bbox)
        self.time_since_update = 0
        Track.count += 1
        self.id = Track.count
        self.hits = 0
        self.age = 0
        self.cls_name = cls_name

    def update(self, bbox, cls_name):
        self.time_since_update = 0
        self.hits += 1
        self.cls_name = cls_name
        self.kf.update(bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hits = 0
        self.time_since_update += 1
        return x_to_bbox(self.kf.x)

    def get_state(self):
        return x_to_bbox(self.kf.x)


class SORT:
    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.frame_count = 0

    def update(self, detections):
        """detections: list of (bbox[x1,y1,x2,y2], cls_name). Returns list of
        (track_id, bbox, cls_name) for confirmed tracks."""
        self.frame_count += 1

        # Predict new locations for existing tracks
        predicted = []
        for t in self.tracks:
            predicted.append(t.predict()[0])

        # Match detections to tracks via IoU + Hungarian algorithm
        matched, unmatched_dets, unmatched_trks = self._associate(
            [d[0] for d in detections], predicted
        )

        for det_idx, trk_idx in matched:
            self.tracks[trk_idx].update(detections[det_idx][0], detections[det_idx][1])

        for det_idx in unmatched_dets:
            self.tracks.append(Track(detections[det_idx][0], detections[det_idx][1]))

        results = []
        alive_tracks = []
        for t in self.tracks:
            if t.time_since_update < self.max_age:
                alive_tracks.append(t)
                if t.time_since_update == 0 and (t.hits >= self.min_hits or self.frame_count <= self.min_hits):
                    bbox = t.get_state()[0]
                    results.append((t.id, bbox, t.cls_name))
        self.tracks = alive_tracks

        return results

    def _associate(self, det_boxes, trk_boxes):
        if len(trk_boxes) == 0:
            return [], list(range(len(det_boxes))), []
        if len(det_boxes) == 0:
            return [], [], list(range(len(trk_boxes)))

        iou_matrix = np.zeros((len(det_boxes), len(trk_boxes)), dtype=np.float32)
        for d, det in enumerate(det_boxes):
            for t, trk in enumerate(trk_boxes):
                iou_matrix[d, t] = iou(det, trk)

        row_idx, col_idx = linear_sum_assignment(-iou_matrix)

        matched, unmatched_dets, unmatched_trks = [], [], []
        for d in range(len(det_boxes)):
            if d not in row_idx:
                unmatched_dets.append(d)
        for t in range(len(trk_boxes)):
            if t not in col_idx:
                unmatched_trks.append(t)

        for d, t in zip(row_idx, col_idx):
            if iou_matrix[d, t] < self.iou_threshold:
                unmatched_dets.append(d)
                unmatched_trks.append(t)
            else:
                matched.append((d, t))

        return matched, unmatched_dets, unmatched_trks


# ---------------------------------------------------------------------------
# Main detection + tracking loop
# ---------------------------------------------------------------------------
def get_color(track_id):
    np.random.seed(track_id)
    return tuple(int(c) for c in np.random.randint(0, 255, 3))


def main():
    parser = argparse.ArgumentParser(description="Real-time object detection and tracking.")
    parser.add_argument("--source", default="0", help="Webcam index (e.g. 0) or path to video file")
    parser.add_argument("--model", default="yolov8n.pt", help="Pre-trained YOLO weights")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--output", default=None, help="Optional path to save annotated output video")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    print("Loading YOLO model (auto-downloads weights on first run)...")
    model = YOLO(args.model)
    class_names = model.names

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    tracker = SORT(max_age=15, min_hits=3, iou_threshold=0.3)

    print("Running. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detect objects in this frame
        results = model.predict(frame, conf=args.conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            cls_name = class_names.get(cls_id, str(cls_id))
            detections.append((np.array([x1, y1, x2, y2]), cls_name))

        # 2. Update tracker with this frame's detections
        tracks = tracker.update(detections)

        # 3. Draw bounding boxes + labels + tracking IDs
        for track_id, bbox, cls_name in tracks:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = get_color(track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} ID:{track_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

        cv2.imshow("Object Detection & Tracking", frame)
        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
