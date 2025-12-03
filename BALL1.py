import cv2
from ultralytics import YOLO

def live_yolov11_detection(model_path="yolo11n.pt", conf_thresh=0.4):
    """
    Real-time YOLOv11 detection that only detects 'person' and 'sports ball',
    and renames them as 'player' and 'volleyball' respectively.
    """
    # Load YOLOv11 model
    print("[INFO] Loading YOLOv11 model...")
    model = YOLO(model_path)

    # Print all class names
    print("\n[INFO] Model Class Labels:")
    for i, name in model.names.items():
        print(f"{i}: {name}")
    print()

    # Get class IDs for 'person' and 'sports ball'
    target_classes = [i for i, name in model.names.items() if name in ["person", "sports ball"]]
    print(f"[INFO] Detecting only: {[model.names[i] for i in target_classes]}")

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera.")
        return

    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame.")
            break

        # Run YOLOv11 inference only for selected classes
        results = model(frame, conf=conf_thresh, imgsz=640, classes=target_classes)

        # Get detections for frame
        annotated_frame = frame.copy()

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Rename classes
                label = model.names[cls_id]
                if label == "person":
                    label = "player"
                elif label == "sports ball":
                    label = "volleyball"

                # Draw bounding box and label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    f"{label} ({conf:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        # Display output
        cv2.imshow("YOLOv11 - Volleyball Detection", annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release camera
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera closed.")

if __name__ == "__main__":
    live_yolov11_detection("yolo11n.pt", conf_thresh=0.4)
