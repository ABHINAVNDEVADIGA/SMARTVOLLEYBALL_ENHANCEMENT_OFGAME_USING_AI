import cv2
from ultralytics import YOLO

def live_yolov11_detection(model_path="yolo11n.pt", conf_thresh=0.4):
    """
    Real-time object detection using YOLOv11 through a live camera.
    :param model_path: path to YOLOv11 model (e.g., yolo11n.pt)
    :param conf_thresh: confidence threshold for detections
    """
    # Load YOLOv11 model
    print("[INFO] Loading YOLOv11 model...")
    model = YOLO(model_path)

    # Start webcam (0 = default webcam)
    print("[INFO] Starting camera...")
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

        # Run YOLOv11 inference
        results = model(frame, conf=conf_thresh, imgsz=640)

        # Get annotated frame
        annotated_frame = results[0].plot()

        # Display
        cv2.imshow("YOLOv11 Live Detection", annotated_frame)

        # Quit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera closed.")

if __name__ == "__main__":
    live_yolov11_detection("yolo11n.pt", conf_thresh=0.4)
