import cv2
import sys
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import ui
import config

LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
POSE_PTS   = [1, 152, 263, 33, 287, 57]

HEAD_3D = np.array([
    [ 0.0,    0.0,    0.0  ],
    [ 0.0,  -63.6,  -12.5 ],
    [-43.3,  32.7,  -26.0 ],
    [ 43.3,  32.7,  -26.0 ],
    [-28.9, -28.9,  -24.1 ],
    [ 28.9, -28.9,  -24.1 ],
], dtype=np.float64)

_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
_closed = 0

def _ear(pts):
    v = distance.euclidean(pts[1], pts[5]) + distance.euclidean(pts[2], pts[4])
    return v / (2.0 * distance.euclidean(pts[0], pts[3]))

def _eye_pts(lm, idx, w, h):
    return np.array([(lm[i].x * w, lm[i].y * h) for i in idx])

def _head_pose(lm, w, h):
    pts2d = np.array([(lm[i].x * w, lm[i].y * h) for i in POSE_PTS], dtype=np.float64)
    cam   = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(HEAD_3D, pts2d, cam, np.zeros((4,1)),
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(
        np.hstack((rmat, np.zeros((3, 1))))
    )
    return float(euler[1]), float(euler[0])

def _draw(frame, ear_val, yaw, pitch, drowsy, distracted, left, right):
    w = frame.shape[1]
    ui.eye_dots(frame, np.concatenate([left, right]))
    ui.metric_card(frame, w-140, 10, 130, 42, "EAR", f"{ear_val:.2f}",
                   value_color=(3, 60, 200) if drowsy else ui.ORANGE)
    ui.metric_card(frame, w-140, 58,  62, 42, "YAW",   f"{yaw:+.0f}")
    ui.metric_card(frame, w-72,  58,  62, 42, "PITCH", f"{pitch:+.0f}")
    ui.bar(frame, w-140, 108, 130, 8, min(ear_val/0.4, 1.0),
           color=(3, 60, 200) if drowsy else ui.ORANGE)
    if drowsy:
        ui.alert_banner(frame, "SONOLENCIA DETECTADA", color=(40, 40, 180))
    elif distracted:
        ui.alert_banner(frame, "DISTRAIDO", color=(3, 140, 200))

def run(frame):
    global _closed
    h, w = frame.shape[:2]
    res  = _mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    out = {"face_found": False, "ear": 0.0, "yaw": 0.0,
           "pitch": 0.0, "drowsy": False, "distracted": False}

    if not res.multi_face_landmarks:
        ui.no_face(frame)
        return frame, out

    lm  = res.multi_face_landmarks[0].landmark
    left  = _eye_pts(lm, LEFT_EYE,  w, h)
    right = _eye_pts(lm, RIGHT_EYE, w, h)
    ear   = (_ear(left) + _ear(right)) / 2.0

    _closed = _closed + 1 if ear < config.EAR_THRESHOLD else 0

    yaw, pitch = _head_pose(lm, w, h)

    out.update({
        "face_found": True,
        "ear":        ear,
        "yaw":        yaw,
        "pitch":      pitch,
        "drowsy":     _closed >= config.EAR_CONSEC_FRAMES,
        "distracted": abs(yaw) > config.HEAD_POSE_YAW_THRESHOLD or
                      abs(pitch) > config.HEAD_POSE_PITCH_THRESHOLD,
    })

    _draw(frame, ear, yaw, pitch, out["drowsy"], out["distracted"], left, right)
    return frame, out