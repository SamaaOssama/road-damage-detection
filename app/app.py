
import tempfile
import time
from pathlib import Path

import cv2
import gradio as gr
import pandas as pd
from ultralytics import YOLO

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "best_model.pt"
NAME_FILE = APP_DIR / "best_model.txt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"{MODEL_PATH.name} not found. Put it next to app.py")

model = YOLO(str(MODEL_PATH))
CLASS_NAMES = model.names                      # {id: name}
ALL_CLASSES = list(CLASS_NAMES.values())
MODEL_LABEL = NAME_FILE.read_text().strip() if NAME_FILE.exists() else MODEL_PATH.name
COLUMNS = ["class", "confidence", "x1", "y1", "x2", "y2"]


def _class_ids(selected):
    ids = [i for i, n in CLASS_NAMES.items() if n in (selected or [])]
    return ids or None


def detect_image(image, conf, iou, selected):
    """image: PIL.Image -> (annotated RGB array, detections table, summary text)."""
    if image is None:
        return None, pd.DataFrame(columns=COLUMNS), "Upload an image first."
    t0 = time.perf_counter()
    r = model.predict(image, conf=conf, iou=iou, classes=_class_ids(selected),
                      imgsz=640, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000

    annotated = r.plot()[:, :, ::-1]          # BGR -> RGB
    rows = [
        [CLASS_NAMES[int(c)], round(float(s), 3), *[round(float(v), 1) for v in box]]
        for c, s, box in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())
    ]
    table = pd.DataFrame(rows, columns=COLUMNS)

    if len(table):
        counts = ", ".join(f"{k}: {v}" for k, v in table["class"].value_counts().items())
        summary = f"**{len(table)} detections** ({counts})  \nInference: {ms:.0f} ms · {1000 / ms:.1f} FPS"
    else:
        summary = f"No road damage detected.  \nInference: {ms:.0f} ms"
    return annotated, table, summary


def detect_video(video_path, conf, iou, selected, skip, progress=gr.Progress()):
    """Process a video file, return annotated video + per-class counts."""
    if not video_path:
        return None, "Upload a video first."
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in / skip, (w, h))

    counts, times, idx = {}, [], 0
    ids = _class_ids(selected)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % skip:
            continue
        t0 = time.perf_counter()
        r = model.predict(frame, conf=conf, iou=iou, classes=ids, imgsz=640, verbose=False)[0]
        times.append(time.perf_counter() - t0)
        writer.write(r.plot())
        for c in r.boxes.cls.tolist():
            counts[CLASS_NAMES[int(c)]] = counts.get(CLASS_NAMES[int(c)], 0) + 1
        progress(idx / total, desc=f"Frame {idx}/{total}")
    cap.release()
    writer.release()

    avg_fps = len(times) / sum(times) if times else 0
    lines = [f"Processed {len(times)} frames · average model speed {avg_fps:.1f} FPS"]
    lines += [f"- {k}: {v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])]
    return out_path, "\n".join(lines)


# ---------------- UI ----------------
with gr.Blocks(title="Road Damage Detection") as demo:
    gr.Markdown(
        f"# 🛣️ Road Damage Detection\n"
        f"Detects **{', '.join(ALL_CLASSES)}** using a YOLO model (`{MODEL_LABEL}`)."
    )

    with gr.Row():
        conf = gr.Slider(0.05, 0.95, value=0.15, step=0.05, label="Confidence threshold")
        iou = gr.Slider(0.10, 0.90, value=0.45, step=0.05, label="NMS IoU threshold")
    classes = gr.CheckboxGroup(ALL_CLASSES, value=ALL_CLASSES, label="Classes to detect")

    with gr.Tab("Image"):
        with gr.Row():
            img_in = gr.Image(type="pil", sources=["upload", "webcam", "clipboard"], label="Road image")
            img_out = gr.Image(type="numpy", label="Detections")
        img_btn = gr.Button("Detect", variant="primary")
        img_summary = gr.Markdown()
        img_table = gr.Dataframe(headers=COLUMNS, label="Detections", interactive=False)

        img_btn.click(detect_image, [img_in, conf, iou, classes], [img_out, img_table, img_summary])
        img_in.upload(detect_image, [img_in, conf, iou, classes], [img_out, img_table, img_summary])

        examples = sorted(p for p in (APP_DIR / "examples").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) \
            if (APP_DIR / "examples").exists() else []
        if examples:
            gr.Examples([[str(p)] for p in examples], inputs=img_in)

    with gr.Tab("Video"):
        with gr.Row():
            vid_in = gr.Video(label="Road video")
            vid_out = gr.Video(label="Annotated video", format="mp4")
        skip = gr.Slider(1, 10, value=2, step=1, label="Process every N-th frame (higher = faster)")
        vid_btn = gr.Button("Run detection", variant="primary")
        vid_summary = gr.Markdown()
        vid_btn.click(detect_video, [vid_in, conf, iou, classes, skip], [vid_out, vid_summary])

    with gr.Accordion("Model info", open=False):
        gr.Markdown(
            f"- **Model:** {MODEL_LABEL}\n"
            f"- **Parameters:** {sum(p.numel() for p in model.model.parameters()) / 1e6:.2f} M\n"
            f"- **Input size:** 640×640\n"
            f"- **Classes:** {', '.join(ALL_CLASSES)}"
        )

if __name__ == "__main__":
    demo.launch()
