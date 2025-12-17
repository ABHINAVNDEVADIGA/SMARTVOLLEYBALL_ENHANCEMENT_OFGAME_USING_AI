# 🏐 Smart Volleyball Referee System BY ABHINAV N DEVADIGA 
### AI-Powered Volleyball Referee Dashboard (YOLO + Streamlit + GPU)
---
# 🎬 Dashboard Demo
### 🔹 REFREE DEMO

![DASHBOARD](media/pic1.png) ![DASHBOARD](media/pic2.png) 

# 🎬 Demo Output Videos

### 🔹 Ball Detection  

![Ball Detection](media/balldetection.gif)  ![Ball Detection](media/y7Detect_volleyball15.gif)

### 🔹 Court Dection

![Detecting Court](media/court.gif)

### 🔹 Player Dection

![Player Detection](media/players_men.gif)

### 🔹 Action Dection

![Action Detection](media/sliding_window.gif)



This project is a fully functional **AI Referee System for Volleyball**, built using **YOLO (Ultralytics)**, **OpenCV**, and a real-time **Streamlit Dashboard**.

The system can:

- 🟡 Detect the **volleyball**
- 🔵 Detect **player actions** (spike, block, serve, receive, set)
- 🟢 Detect the **court boundary**
- 🎯 Decide if the ball is **IN** or **OUT**
- 🏆 Automatically award points
- ⚡ Run in real-time with an NVIDIA GPU (RTX 3050 recommended)

You can use:
- 💻 Laptop Webcam
- 📱 Mobile Phone Camera (IP Webcam / DroidCam)
- 📂 Any uploaded match video

All inside a clean, interactive dashboard.

---

# 🚀 Features

### 🔍 AI Detection Features
- Ball detection  
- Player action detection  
- Court detection  
- IN/OUT logic  
- Auto score decision  
- HD and SD inference modes  
- GPU-accelerated inference  

### 🖥 Dashboard Features
- Laptop webcam input  
- Mobile camera input (manual URL + auto-detect)  
- Video upload  
- FPS counter  
- Latency counter  
- Live “Connected / Disconnected” indicator  
- Live preview + detection window  
- Fixed two-column layout  
- Clean UI designed for real-time use  

---

# 🧠 Tech Stack Used

- **Python**
- **Streamlit**
- **OpenCV**
- **Ultralytics YOLOv8 / YOLO11**
- **PyTorch (with CUDA 12.x GPU acceleration)**
- **Requests library**

Runs best on:
- ✔ NVIDIA RTX 3050 or higher  
- ✔ Any CUDA-enabled GPU


if using python environ ment   C:\projects\VOLLEYBAL> venv\Scripts\activate
streamlit run dashboard.py

or directly run from Run_Dashboard.py
