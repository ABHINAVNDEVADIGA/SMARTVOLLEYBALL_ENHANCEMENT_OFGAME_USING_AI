from ultralytics import YOLO
import cv2

# Load YOLO model
print("[INFO] Loading YOLO model...")
model = YOLO('best.pt')  # Replace with your model path
print("[INFO] Model loaded successfully!")

# Print all class labels the model can detect
print("\n[INFO] Classes that the model can detect:")
for idx, name in model.names.items():
    print(f"  ID {idx}: {name}")
print("----------------------------------------\n")

# ---------- Live Webcam Detection ----------
def live_detection():
    print("[INFO] Starting live detection...")
    cap = cv2.VideoCapture("rally_men.mp4")

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return
    else:
        print("[INFO] Webcam opened successfully.")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame not received. Exiting loop...")
            break

        frame_count += 1
        print(f"\n[FRAME {frame_count}] Running detection...")

        results = model(frame)
        annotated_frame = results[0].plot()

        # Get detected labels
        boxes = results[0].boxes
        class_ids = boxes.cls.tolist() if boxes is not None else []
        detected_labels = [model.names[int(cls_id)] for cls_id in class_ids]

        if detected_labels:
            print(f"[DETECTED OBJECTS] {detected_labels}")
        else:
            print("[INFO] No objects detected in this frame.")

        cv2.imshow("YOLOv8 - Live Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[ACTION] Quit key pressed. Exiting detection loop...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam released and windows closed. Detection finished.")


# ---------- Run Live Detection ----------
if __name__ == "__main__":
    print("[SYSTEM] Running in TEST MODE (Live Detection Only)")
    live_detection()
