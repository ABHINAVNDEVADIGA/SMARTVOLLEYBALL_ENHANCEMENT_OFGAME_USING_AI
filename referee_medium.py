import cv2
from ultralytics import YOLO
import torch
import time

# ============================================================
#                GPU (RTX 3050)
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] Using device:", device)

# ============================================================
#               LOAD MODELS ON GPU (FP16)
# ============================================================
ball_model = YOLO("weights/ball/weights/best.pt").to(device)
action_model = YOLO("weights/action/weights/action.pt").to(device)
court_model = YOLO("weights/court/weights/court.pt").to(device)

print("[INFO] Models loaded on RTX 3050 GPU.")

# ============================================================
#               OPEN FULL-HD VIDEO
# ============================================================
cap = cv2.VideoCapture("weights/action/weights/rally_men.mp4")

if not cap.isOpened():
    print("[ERROR] Cannot open video.")
    exit()

teamA, teamB = 0, 0
frame_id = 0
last_touch = None
ball_in = True

print("[INFO] Running referee system...")

# ============================================================
#              MAIN LOOP
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    H, W = frame.shape[:2]
    start = time.time()

    # ---------------- SMALL DISPLAY WINDOW ----------------
    display = cv2.resize(frame, (640, 360))
    scale_x = 640 / W
    scale_y = 360 / H

    # ============================================================
    # 1️⃣ BALL DETECTION (EVERY FRAME)
    # ============================================================
    ball_results = ball_model.predict(
        frame, imgsz=1280, conf=0.25, half=True, device=device, verbose=False
    )
    ball_boxes = ball_results[0].boxes

    ball_xy = None
    if ball_boxes is not None and len(ball_boxes) > 0:
        box = ball_boxes[0]
        bx1, by1, bx2, by2 = map(int, box.xyxy[0])

        # Ball center
        ball_xy = ((bx1 + bx2)//2, (by1 + by2)//2)

        # DRAW BALL BOX (scaled to display)
        sx1 = int(bx1 * scale_x)
        sy1 = int(by1 * scale_y)
        sx2 = int(bx2 * scale_x)
        sy2 = int(by2 * scale_y)

        cv2.rectangle(display, (sx1, sy1), (sx2, sy2), (0,255,255), 2)
        cv2.putText(display, "BALL", (sx1, sy1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # ============================================================
    # 2️⃣ ACTION DETECTION (EVERY 2 FRAMES)
    # ============================================================
    detected_action = None

    if frame_id % 2 == 0:
        act_results = action_model.predict(
            frame, imgsz=1280, conf=0.25, half=True, device=device, verbose=False
        )
        act_boxes = act_results[0].boxes

        if act_boxes is not None:
            for box in act_boxes:
                cid = int(box.cls[0])

                # Skip ball class
                if cid == 0:
                    continue

                detected_action = action_model.names[cid]

                # Draw action box on the PLAYER
                ax1, ay1, ax2, ay2 = map(int, box.xyxy[0])

                sx1 = int(ax1 * scale_x)
                sy1 = int(ay1 * scale_y)
                sx2 = int(ax2 * scale_x)
                sy2 = int(ay2 * scale_y)

                cv2.rectangle(display, (sx1, sy1), (sx2, sy2), (255,0,0), 2)
                cv2.putText(display, detected_action.upper(), (sx1, sy1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

                break

    # ============================================================
    # 3️⃣ COURT DETECTION (EVERY 4 FRAMES)
    # ============================================================
    if frame_id % 4 == 0:
        court_results = court_model.predict(
            frame, imgsz=1280, conf=0.25, half=True, device=device, verbose=False
        )
        court_boxes = court_results[0].boxes

        if court_boxes is not None and len(court_boxes) > 0:
            cx1, cy1, cx2, cy2 = map(int, court_boxes[0].xyxy[0])

            # DRAW COURT BOX
            sx1 = int(cx1 * scale_x)
            sy1 = int(cy1 * scale_y)
            sx2 = int(cx2 * scale_x)
            sy2 = int(cy2 * scale_y)

            cv2.rectangle(display, (sx1, sy1), (sx2, sy2), (0,255,0), 2)
            cv2.putText(display, "COURT", (sx1, sy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # Ball in/out
            if ball_xy:
                ball_in = cx1 < ball_xy[0] < cx2 and cy1 < ball_xy[1] < cy2

    # ============================================================
    # 4️⃣ LAST TOUCH + POINT LOGIC
    # ============================================================
    if ball_xy:
        last_touch = "A" if ball_xy[0] < W//2 else "B"

    if not ball_in and last_touch:
        if last_touch == "A":
            teamB += 1
        else:
            teamA += 1
        ball_in = True
        last_touch = None

    # ============================================================
    # 5️⃣ SCOREBOARD + FPS
    # ============================================================
    fps = int(1 / (time.time() - start))

    cv2.putText(display, f"Team A: {teamA}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
    cv2.putText(display, f"Team B: {teamB}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
    cv2.putText(display, f"FPS: {fps}", (500,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    if detected_action:
        cv2.putText(display, f"ACTION: {detected_action.upper()}",
                    (20, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    cv2.imshow("Referee (GPU HD Accuracy)", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
