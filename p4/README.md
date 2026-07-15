# VISION // Object Recognition Pipeline

An interactive computer-vision instrument that detects and localizes objects in an image using a pre-trained **MobileNet-SSD** network, wrapped in a live Streamlit interface that exposes every stage of the pipeline — not just the final result.

Built for **AI Project 4: Image or Text Recognition (Basic)** — DecodeLabs Industrial Training Kit, Batch 2026.
**Path chosen: Path 2 — Object Detection.**

---

## What it does

Upload an image (or use a sample) and the app runs it through a real six-stage recognition pipeline:

```
01 Input Acquisition  →  02 Preprocessing  →  03 Blob Construction
        →  04 Neural Inference  →  05 Confidence Filter  →  06 Render Output
```

Each stage is timed independently and shown live. The output is a set of bounding boxes, class labels, and confidence percentages — exactly what the model predicted, filtered against a mandatory 80% confidence floor.

## Why MobileNet-SSD (transfer learning, not training from scratch)

The model ships pre-trained on **VOC0712** (20 object classes, mAP 0.727). Rather than training a detector from millions of images ourselves, this project reuses a network that already learned edges, shapes, and textures, and applies it directly — the transfer-learning approach the assignment specifies. MobileNet-SSD's depthwise separable convolutions factor a standard convolution into a per-channel spatial pass and a 1×1 pointwise combination, cutting compute roughly 8–9× versus an equivalent standard convolution, which is what makes it viable for real-time / edge inference instead of a heavier detector like Faster R-CNN.

## Pipeline detail

| Stage | What happens | Why it's there |
|---|---|---|
| **Preprocessing** | Gaussian blur (noise reduction) + CLAHE adaptive contrast enhancement on the luminance channel only | Suppresses sensor/compression noise and boosts local contrast in poorly-lit regions without distorting colour — the colour-domain analogue of adaptive thresholding, since a full grayscale conversion would destroy the colour signal the detector uses |
| **Blob construction** | `cv2.dnn.blobFromImage` resizes to 300×300 and normalizes pixel values | This is the exact input shape and scale MobileNet-SSD was trained on |
| **Neural inference** | Forward pass through the Caffe model via `cv2.dnn` | Produces per-candidate class scores and normalized box coordinates |
| **Confidence filtering** | Every prediction below **80%** is discarded, never rendered | Mandatory accuracy benchmark from the project spec — enforced in code, not just the UI (the slider cannot go below 0.80) |
| **Render** | Bounding boxes, class labels, and confidence percentages drawn onto the image; each class gets a deterministic colour | Visual confirmation requirement — results must be legible at a glance |

## Project structure

```
vision-project/
├── app.py                          # Streamlit interface
├── core/
│   ├── preprocess.py                # noise reduction + adaptive contrast
│   └── detector.py                  # blob construction, inference, filtering, drawing
├── model/
│   ├── MobileNetSSD_deploy.prototxt # network architecture
│   └── MobileNetSSD_deploy.caffemodel  # pre-trained weights (VOC0712)
├── sample_images/                   # demo images for quick testing
├── requirements.txt
└── README.md
```

## Known limitation: closed-set classification

MobileNet-SSD can only ever output one of its 20 trained classes (or background) — it has no "unknown object" option. On an image that doesn't clearly match any of those 20 categories (e.g. an extreme close-up crop with heavy motion blur, no visible face or limbs), the model is still forced to pick its best-scoring option among the 20 it knows, and can report high confidence in a wrong answer — for example, classifying a blurred close-up of skin as "bottle" at 93%.

This is expected behavior for any closed-set classifier, not a bug in this pipeline. The confidence score measures how strongly one known class beat the other 19 — it is not an absolute measure of correctness, and it says nothing about whether the true object is outside the model's 20-class vocabulary at all. Mitigating this properly would require either a model trained with an explicit "unknown/other" class, or an out-of-distribution detection layer on top — both out of scope for this basic-tier project, which is limited to using an existing pre-trained model rather than retraining one.

## Known issue: OpenCV 5.0

OpenCV 5.0 (released June 2026) removed `cv2.dnn.readNetFromCaffe()` and `readNetFromDarknet()` entirely — Caffe/Darknet model loading no longer exists in the new engine. `requirements.txt` pins `opencv-python<5.0` for this reason. If you ever see `AttributeError: module 'cv2.dnn' has no attribute 'readNetFromCaffe'`, your environment has OpenCV 5.x installed; fix with:

```bash
pip install "opencv-python>=4.9,<5.0" --force-reinstall
```

## Setup & run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (defaults to `http://localhost:8501`).

## How this satisfies the four validation requirements

1. **Library integration** — `cv2.dnn` loads and runs the pre-trained MobileNet-SSD Caffe model (`core/detector.py`).
2. **Preprocessing integrity** — Gaussian noise reduction and CLAHE adaptive contrast enhancement run before every inference (`core/preprocess.py`), visibly toggleable in the UI.
3. **Accuracy benchmark** — 80% is the hard floor in code (`MANDATORY_MIN_CONFIDENCE` in `core/detector.py`); the UI slider cannot be set below it.
4. **Visual confirmation** — bounding boxes, labels, and confidence percentages are rendered directly on the image and listed with per-object confidence bars and pixel coordinates.

## Design notes

The interface is deliberately built like a machine-vision instrument rather than a marketing page: the detection canvas itself is the centerpiece, confidence values are shown as raw telemetry (JetBrains Mono, percentages, pixel coordinates), and the pipeline stepper reflects a real, ordered sequence of operations rather than decorative numbering.
