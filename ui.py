import cv2

# BGR
ORANGE = (3,  102, 239)
BG     = (64,  40,  37)
TEXT   = (240, 233, 232)
MUTED  = (130, 120, 110)
BORDER = (92,  61,  58)

def pill(frame, text, x, y, bg=ORANGE, fg=TEXT, scale=0.45):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    px, py = 8, 5
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + tw + px*2, y + th + py*2), bg, -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    cv2.putText(frame, text, (x + px, y + th + py - 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 1, cv2.LINE_AA)

def bar(frame, x, y, w, h, ratio, color=ORANGE, label=None):
    ratio = max(0.0, min(1.0, ratio))
    cv2.rectangle(frame, (x, y), (x + w, y + h), BG, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), BORDER, 1)
    if ratio > 0:
        cv2.rectangle(frame, (x, y), (x + int(w * ratio), y + h), color, -1)
    if label:
        cv2.putText(frame, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, MUTED, 1, cv2.LINE_AA)

def alert_banner(frame, text, color=ORANGE):
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    bx = w // 2 - tw // 2 - 14
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, 14), (bx + tw + 28, th + 32), color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (bx, 14), (bx + tw + 28, th + 32), color, 1)
    cv2.putText(frame, text, (bx + 14, th + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEXT, 2, cv2.LINE_AA)

def hud_box(frame, x, y, w, h, label=None, color=ORANGE):
    # cantos estilo HUD
    s, t = 14, 2
    pts = [
        ((x,     y),     (x+s,   y),   (x,   y+s)),
        ((x+w,   y),     (x+w-s, y),   (x+w, y+s)),
        ((x,     y+h),   (x+s,   y+h), (x,   y+h-s)),
        ((x+w,   y+h),   (x+w-s, y+h), (x+w, y+h-s)),
    ]
    for corner, he, ve in pts:
        cv2.line(frame, corner, he, color, t, cv2.LINE_AA)
        cv2.line(frame, corner, ve, color, t, cv2.LINE_AA)
    if label:
        pill(frame, label, x, y - 26, bg=color)

def eye_dots(frame, points, color=ORANGE):
    for pt in points:
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, color, -1, cv2.LINE_AA)

def metric_card(frame, x, y, w, h, title, value, value_color=ORANGE):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), BG, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), BORDER, 1)
    cv2.putText(frame, title, (x + 8, y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, MUTED, 1, cv2.LINE_AA)
    cv2.putText(frame, value, (x + 8, y + h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, value_color, 1, cv2.LINE_AA)

def no_face(frame):
    h = frame.shape[0]
    cv2.putText(frame, "rosto nao detectado", (12, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, MUTED, 1, cv2.LINE_AA)