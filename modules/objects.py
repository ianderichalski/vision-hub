import sys
import logging
from ultralytics import YOLO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import ui
import config

logger = logging.getLogger(__name__)

TRACKED = {
    "cell phone", "laptop", "book", "cup", "bottle",
    "keyboard", "mouse", "remote", "backpack",
}

_model      = None
_last_boxes = []

def _load():
    global _model
    if _model is not None:
        return True
    try:
        _model = YOLO("yolov8n.pt")
        return True
    except Exception as e:
        logger.error(f"falha ao carregar YOLO: {e}")
        return False

def _draw(frame, boxes):
    for b in boxes:
        x1, y1, x2, y2 = b["coords"]
        ui.hud_box(frame, x1, y1, x2-x1, y2-y1,
                   label=f"{b['label']}  {b['conf']:.0%}")
    if boxes:
        ui.pill(frame, f"{len(boxes)} objeto(s)", 12, frame.shape[0] - 38)

def run(frame, process=True):
    global _last_boxes

    if not _load():
        ui.pill(frame, "modelo não carregado", 12, 12, bg=config.RED)
        return frame, {"objects": []}

    if process:
        _last_boxes = [
            {"label": _model.names[int(b.cls[0])],
             "conf":  float(b.conf[0]),
             "coords": tuple(map(int, b.xyxy[0]))}
            for b in _model(frame, verbose=False)[0].boxes
            if float(b.conf[0]) >= config.OBJECT_CONFIDENCE
            and _model.names[int(b.cls[0])] in TRACKED
        ]

    _draw(frame, _last_boxes)
    return frame, {"objects": _last_boxes}