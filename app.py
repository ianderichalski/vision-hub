import cv2
import sys
import logging
import numpy as np
import pygame
import config
from modules import focus, emotion, objects

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODULES = {"focus": False, "emotion": False, "objects": False}

SIDEBAR = [
    {"key": "focus",   "label": "Foco",    "sub": "sonolência + atenção", "k": pygame.K_1},
    {"key": "emotion", "label": "Emoção",  "sub": "reconhecimento",       "k": pygame.K_2},
    {"key": "objects", "label": "Objetos", "sub": "detecção YOLO",        "k": pygame.K_3},
]

BG1    = (47,  50,  77)
BG2    = (37,  40,  64)
ORANGE = (239, 102,   3)
TEXT   = (232, 233, 240)
MUTED  = (107, 107, 111)
BORDER = (58,   61,  92)
SW     = config.SIDEBAR_W

def draw_sidebar(surf, fonts):
    ft, fl, fs, fx = fonts
    h = surf.get_height()

    pygame.draw.rect(surf, BG2, (0, 0, SW, h))
    pygame.draw.line(surf, BORDER, (SW, 0), (SW, h), 1)

    t1 = ft.render("VISION", True, ORANGE)
    surf.blit(t1, (16, 18))
    surf.blit(ft.render("HUB", True, TEXT), (16 + t1.get_width() + 4, 18))
    surf.blit(fx.render("IA em tempo real", True, MUTED), (16, 44))
    pygame.draw.line(surf, BORDER, (14, 62), (SW - 14, 62), 1)

    y = 78
    for i, m in enumerate(SIDEBAR):
        active = MODULES[m["key"]]
        rect   = (10, y, SW - 20, 52)

        if active:
            pygame.draw.rect(surf, ORANGE, rect, border_radius=8)
            tc, sc, kc, dc = TEXT, (235, 215, 195), (255, 255, 255), TEXT
        else:
            pygame.draw.rect(surf, BG1,    rect, border_radius=8)
            pygame.draw.rect(surf, BORDER, rect, width=1, border_radius=8)
            tc, sc, kc, dc = TEXT, (160, 163, 190), (200, 180, 100), ORANGE

        pygame.draw.circle(surf, dc, (28, y + 18), 5)
        surf.blit(fl.render(m["label"], True, tc), (42, y + 8))
        surf.blit(fs.render(m["sub"],   True, sc), (42, y + 28))
        surf.blit(fx.render(str(i+1),   True, kc), (SW - 22, y + 14))
        y += 62

    pygame.draw.line(surf, BORDER, (14, h - 38), (SW - 14, h - 38), 1)
    surf.blit(fx.render("Q  —  sair", True, (160, 163, 190)), (16, h - 26))

def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        logger.error("webcam nao encontrada")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    pygame.init()
    screen = pygame.display.set_mode((config.FRAME_WIDTH + SW, config.FRAME_HEIGHT))
    pygame.display.set_caption("VisionHub")
    clock  = pygame.time.Clock()

    fonts = (
        pygame.font.SysFont("segoeui", 20, bold=True),
        pygame.font.SysFont("segoeui", 15, bold=True),
        pygame.font.SysFont("segoeui", 12),
        pygame.font.SysFont("segoeui", 11),
    )

    n = 0
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                cap.release(); pygame.quit(); sys.exit(0)
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_q:
                    cap.release(); pygame.quit(); sys.exit(0)
                for m in SIDEBAR:
                    if e.key == m["k"]:
                        MODULES[m["key"]] = not MODULES[m["key"]]
                        logger.info(f"{m['key']} → {'ON' if MODULES[m['key']] else 'OFF'}")

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        n += 1

        if MODULES["focus"]:
            frame, _ = focus.run(frame)
        if MODULES["emotion"]:
            frame, _ = emotion.run(frame, process=n % config.EMOTION_SKIP_FRAMES == 0)
        if MODULES["objects"]:
            frame, _ = objects.run(frame, process=n % config.OBJECT_SKIP_FRAMES == 0)

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))

        screen.fill(BG1)
        screen.blit(surf, (SW, 0))
        draw_sidebar(screen, fonts)

        fps = fonts[3].render(f"{int(clock.get_fps())} fps", True, MUTED)
        screen.blit(fps, (screen.get_width() - fps.get_width() - 10, screen.get_height() - 22))

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()