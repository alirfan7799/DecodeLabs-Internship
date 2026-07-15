import time
from dataclasses import dataclass, field

import cv2
import numpy as np

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]

_rng = np.random.default_rng(42)
CLASS_COLORS = {cls: tuple(int(c) for c in _rng.integers(60, 255, 3)) for cls in CLASSES}

BLOB_SIZE = (300, 300)
BLOB_SCALE = 0.007843  # 1/127.5
BLOB_MEAN = 127.5

MANDATORY_MIN_CONFIDENCE = 0.80


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple  # (x1, y1, x2, y2) in pixel coordinates


@dataclass
class InferenceResult:
    detections: list = field(default_factory=list)
    rejected_count: int = 0
    stage_timings_ms: dict = field(default_factory=dict)
    annotated_image: np.ndarray = None


class ObjectDetector:

    # Load the pre-trained Caffe model
    def __init__(self, prototxt_path: str, model_path: str):
        if not hasattr(cv2.dnn, "readNetFromCaffe"):
            raise RuntimeError(
                f"This OpenCV build (cv2 {cv2.__version__}) has no cv2.dnn.readNetFromCaffe — "
                "OpenCV 5.0 removed Caffe/Darknet model loading entirely. "
                'Fix: pip install "opencv-python>=4.9,<5.0" --force-reinstall'
            )
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Resize and normalize image to 300x300 blob
    def _build_blob(self, image: np.ndarray) -> np.ndarray:
        return cv2.dnn.blobFromImage(
            cv2.resize(image, BLOB_SIZE), BLOB_SCALE, BLOB_SIZE, BLOB_MEAN
        )

    # Run the full detection pipeline (preprocess -> infer -> filter)
    def infer(self, image: np.ndarray, confidence_threshold: float = MANDATORY_MIN_CONFIDENCE) -> InferenceResult:
        confidence_threshold = max(confidence_threshold, MANDATORY_MIN_CONFIDENCE)
        (h, w) = image.shape[:2]
        timings = {}

        t0 = time.perf_counter()
        blob = self._build_blob(image)
        timings["blob_construction_ms"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        self.net.setInput(blob)
        raw = self.net.forward()
        timings["inference_ms"] = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        detections = []
        rejected = 0
        for i in range(raw.shape[2]):
            confidence = float(raw[0, 0, i, 2])
            class_idx = int(raw[0, 0, i, 1])
            if class_idx >= len(CLASSES):
                continue
            if confidence < confidence_threshold:
                if confidence > 0.01:  # ignore low-confidence noise
                    rejected += 1
                continue
            box = raw[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            detections.append(Detection(label=CLASSES[class_idx], confidence=confidence, box=(x1, y1, x2, y2)))
        timings["filtering_ms"] = (time.perf_counter() - t2) * 1000

        detections.sort(key=lambda d: d.confidence, reverse=True)

        t3 = time.perf_counter()
        annotated = self._draw(image, detections)
        timings["render_ms"] = (time.perf_counter() - t3) * 1000

        return InferenceResult(
            detections=detections,
            rejected_count=rejected,
            stage_timings_ms=timings,
            annotated_image=annotated,
        )

    # Draw bounding boxes and labels on the image
    def _draw(self, image: np.ndarray, detections: list) -> np.ndarray:
        out = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det.box
            color = CLASS_COLORS[det.label]
            thickness = max(2, int(round(min(out.shape[:2]) / 300)))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

            label_text = f"{det.label}: {det.confidence * 100:.1f}%"
            font_scale = max(0.5, min(out.shape[:2]) / 700)
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            label_y = max(y1, th + 8)
            cv2.rectangle(out, (x1, label_y - th - baseline - 4), (x1 + tw + 6, label_y + baseline - 4), color, -1)
            cv2.putText(out, label_text, (x1 + 3, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (15, 15, 15), 2, cv2.LINE_AA)
        return out
