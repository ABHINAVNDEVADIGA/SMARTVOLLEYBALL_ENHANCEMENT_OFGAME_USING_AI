import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import torch
import time
import requests

# -------------------------------------------------------------
# GPU SETUP
# -------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

ball_model = YOLO("weights/ball/weights/best.pt").to(device)
action_model = YOLO("weights/action/weights/action.pt").to(device)
court_model = YOLO("weights/court/weights/court.pt").to(device)

# -------------------------------------------------------------
# STREAMLIT PAGE
# -------------------------------------------------------------
st.set_page_config(page_title="Volleyball Referee", layout="wide")

# -------------------------------------------------------------
# GLOBAL PLACEHOLDERS (Important!)
# -------------------------------------------------------------
status_box = st.empty()
fps_box = st.empty()
latency_box = st.empty()

# Main UI Title
st.markdown("<h1 style='text-align:center;color:orange;'>🏐 Volleyball Referee Dashboard</h1>", unsafe_allow_html=True)

# -------------------------------------------------------------
# LAYOUT COLUMNS
# -------------------------------------------------------------
col_sidebar, col_main = st.columns([1, 3])

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
with col_sidebar:

    st.markdown("### 🎥 Select Input Source")

    mode = st.radio("Choose Input Type:", ["Laptop Webcam", "Mobile Camera", "Upload Video"])

    # Laptop webcam camera index
    cam_index = 0
    if mode == "Laptop Webcam":
        cam_index = st.selectbox("Select Laptop Camera Index", [0,1,2,3], index=0)

    # Mobile camera
    mobile_url = ""
    if mode == "Mobile Camera":
        mobile_url = st.text_input("Enter Mobile Camera URL:", "http://192.168.1.5:8080/video")

        if st.button("🔍 Auto Detect Mobile Camera"):
            found = False
            status_box.info("🔍 Scanning your network...")
            for i in range(1,255):
                test_url = f"http://192.168.1.{i}:8080/video"
                try:
                    r = requests.get(test_url, timeout=0.15)
                    if r.status_code == 200:
                        mobile_url = test_url
                        found = True
                        status_box.success(f"📱 Found Mobile Camera at {mobile_url}")
                        break
                except:
                    pass
            if not found:
                status_box.error("❌ No mobile camera detected")

    # Video upload
    uploaded_path = None
    if mode == "Upload Video":
        uploaded = st.file_uploader("Upload Video", type=["mp4","avi","mov"])
        if uploaded:
            temp = tempfile.NamedTemporaryFile(delete=False)
            temp.write(uploaded.read())
            uploaded_path = temp.name
            status_box.success("🎬 Video Uploaded Successfully")

    # HD/SD toggle
    st.markdown("### 🖼 Resolution")
    resolution = st.selectbox("Resolution Mode", ["SD (640px)", "HD (1280px)"], index=1)
    imgsz = 1280 if resolution == "HD (1280px)" else 640

    # Start button
    start_button = st.button("▶ START REFEREE", use_container_width=True)

# -------------------------------------------------------------
# MAIN DISPLAY AREA
# -------------------------------------------------------------
with col_main:
    st.markdown("### 🎥 Live Preview")
    video_box = st.empty()
    st.markdown("---")
    detect_box = st.empty()

# -------------------------------------------------------------
# REFEREE ENGINE
# -------------------------------------------------------------
def run_referee(video_source):
    try:
        cap = cv2.VideoCapture(video_source)
        status_box.success("🟢 Connected")
    except:
        status_box.error("🔴 Cannot connect")
        return

    frame_id = 0
    teamA = 0
    teamB = 0
    last_touch = None
    ball_in = True

    while True:
        start_t = time.time()

        ret, frame = cap.read()
        if not ret:
            status_box.error("🔴 Disconnected / Video Ended")
            break

        frame_id += 1

        H, W = frame.shape[:2]

        # Preview
        video_box.image(cv2.resize(frame,(640,360)),channels="BGR")

        # Detection View
        display = cv2.resize(frame,(640,360))
        sx = 640 / W
        sy = 360 / H

        # ---------------- BALL DETECTION ----------------
        ball_results = ball_model.predict(frame, imgsz=imgsz, conf=0.25, half=True, device=device, verbose=False)
        ball_boxes = ball_results[0].boxes
        ball_xy = None

        if len(ball_boxes) > 0:
            bx1,by1,bx2,by2 = map(int, ball_boxes[0].xyxy[0])
            ball_xy = ((bx1+bx2)//2, (by1+by2)//2)

            cv2.rectangle(display,(int(bx1*sx),int(by1*sy)),(int(bx2*sx),int(by2*sy)),(0,255,255),2)
            cv2.putText(display,"BALL",(int(bx1*sx),int(by1*sy)-5),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

        # ---------------- ACTION DETECTION ----------------
        act_results = action_model.predict(frame, imgsz=imgsz, conf=0.25, half=True, device=device, verbose=False)
        for abox in act_results[0].boxes:
            act_id = int(abox.cls[0])
            if act_id == 0:  # skip ball
                continue
            ax1,ay1,ax2,ay2 = map(int, abox.xyxy[0])
            name = action_model.names[act_id]
            cv2.rectangle(display,(int(ax1*sx),int(ay1*sy)),(int(ax2*sx),int(ay2*sy)),(255,0,0),2)
            cv2.putText(display,name.upper(),(int(ax1*sx),int(ay1*sy)-5),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
            break

        # ---------------- COURT DETECTION ----------------
        court_results = court_model.predict(frame, imgsz=imgsz, conf=0.25, half=True, device=device, verbose=False)
        cbox = court_results[0].boxes

        if len(cbox) > 0:
            cx1,cy1,cx2,cy2 = map(int, cbox[0].xyxy[0])
            cv2.rectangle(display,(int(cx1*sx),int(cy1*sy)),(int(cx2*sx),int(cy2*sy)),(0,255,0),2)
            if ball_xy:
                ball_in = cx1 < ball_xy[0] < cx2 and cy1 < ball_xy[1] < cy2

        # ---------------- SCORE ----------------
        cv2.putText(display,f"Team A: {teamA}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.putText(display,f"Team B: {teamB}",(10,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        detect_box.image(display, channels="BGR")

        # FPS + LATENCY
        latency = (time.time() - start_t) * 1000
        fps = 1 / (time.time() - start_t)

        fps_box.info(f"FPS: **{fps:.1f}**")
        latency_box.info(f"Latency: **{latency:.1f} ms**")

    cap.release()

# -------------------------------------------------------------
# START BUTTON ACTION
# -------------------------------------------------------------
if start_button:

    if mode == "Laptop Webcam":
        status_box.info("💻 Starting Laptop Webcam...")
        run_referee(cam_index)

    elif mode == "Mobile Camera":
        status_box.info("📱 Connecting to mobile camera...")
        run_referee(mobile_url)

    elif mode == "Upload Video":
        if uploaded_path:
            status_box.info("🎬 Starting uploaded video...")
            run_referee(uploaded_path)
        else:
            status_box.error("❌ Upload a video first")
