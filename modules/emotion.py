import cv2
import sys
import logging
import numpy as np
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import ui
import config

logger = logging.getLogger(__name__)

MODEL_URL  = "https://huggingface.co/onnxmodelzoo/emotion-ferplus-8/resolve/main/emotion-ferplus-8.onnx"
MODEL_PATH = Path("models/emotion-ferplus-8.onnx")
EMOTIONS   = ["neutro", "feliz", "surpreso", "triste", "raiva"]

_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_net         = None
_last        = None
_load_failed = False

def _load():
    global _net, _load_failed
    if _net is not None:
        return True
    if _load_failed:
        return False
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(exist_ok=True)
        logger.info("baixando modelo de emoção...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as e:
            logger.error(f"falha no download: {e}")
            _load_failed = True
            return False
    _net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
    return True

def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def _draw(frame, faces, dominant, scores):
    h, w = frame.shape[:2]
    for (fx, fy, fw, fh) in faces:
        ui.hud_box(frame, fx, fy, fw, fh, label=dominant.upper())

    bx = w - 134
    y  = 14
    for label, score in zip(EMOTIONS, scores):
        color = ui.ORANGE if label == dominant else ui.BORDER
        ui.bar(frame, bx, y + 14, 120, 6, score, color=color,
               label=f"{label}  {score*100:.0f}%")
        y += 28

    conf = scores[EMOTIONS.index(dominant)] * 100
    ui.pill(frame, f"{dominant.upper()}  {conf:.0f}%", 12, h - 38)

def run(frame, process=True):
    global _last

    out = {"face_found": False, "dominant": None, "scores": []}

    if not _load():
        ui.pill(frame, "modelo não carregado", 12, 12, bg=config.RED)
        return frame, out

    if process:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) > 0:
            fx, fy, fw, fh = faces[0]
            roi  = frame[fy:fy+fh, fx:fx+fw]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blob = cv2.dnn.blobFromImage(cv2.resize(gray_roi, (64, 64)), 1.0, (64, 64))
            _net.setInput(blob)
            scores   = _softmax(_net.forward()[0]).tolist()
            dominant = EMOTIONS[int(np.argmax(scores))]
            _last = {"face_found": True, "dominant": dominant,
                     "scores": scores, "faces": faces}

    if _last and _last["face_found"]:
        _draw(frame, _last["faces"], _last["dominant"], _last["scores"])
        return frame, _last

    ui.no_face(frame)
    return frame, out