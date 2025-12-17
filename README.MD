🏐 Smart Volleyball Referee System
By Abhinav N Devadiga
🎬 Dashboard Demo
<div align="center"> <table> <tr> <td style="border:1px solid #ccc; padding:5px;"> <img src="media/pic1.png" width="400"/> </td> <td style="border:1px solid #ccc; padding:5px;"> <img src="media/pic2.png" width="400"/> </td> </tr> </table> </div>
🎥 Demo Output Videos
🔹 Ball Detection
<div align="center"> <table> <tr> <td style="border:1px solid #ccc; padding:5px;"> <img src="media/balldetection.gif" width="400"/> </td> <td style="border:1px solid #ccc; padding:5px;"> <img src="media/y7Detect_volleyball15.gif" width="400"/> </td> </tr> </table> </div>
🔹 Court Detection
<div align="center"> <table> <tr> <td style="border:1px solid #ccc; padding:5px;"> <img src="media/court.gif" width="400"/> </td> </tr> </table> </div>
🔹 Player Detection
<div align="center"> <table> <tr> <td style="border:1px solid #ccc; padding:5px;"> <img src="media/players_men.gif" width="400"/> </td> </tr> </table> </div>
🔹 Action Detection
<div align="center"> <table> <tr> <td style="border:1px solid #ccc; padding:5px;"> <img src="media/sliding_window.gif" width="400"/> </td> </tr> </table> </div>
🧠 AI-Powered Volleyball Referee Dashboard

(YOLO + Streamlit + GPU)

This project is a real-time AI Volleyball Referee System built using Ultralytics YOLO, OpenCV, and Streamlit.

The system can:

🟡 Detect the volleyball

🔵 Detect player actions (spike, block, serve, receive, set)

🟢 Detect the court boundary

🎯 Decide IN / OUT

🏆 Automatically award points

⚡ Run in real-time on GPU (RTX 3050 recommended)

Input sources:

💻 Laptop Webcam

📱 Mobile Camera (IP Webcam / DroidCam)

📂 Uploaded match videos

All inside a clean, interactive dashboard.

🚀 Features
🔍 AI Detection

Ball detection

Player action detection

Court detection

IN / OUT logic

Auto score decision

HD / SD inference modes

GPU-accelerated inference

🖥 Dashboard

Laptop webcam input

Mobile camera input (manual URL + auto detect)

Video upload

FPS counter

Latency counter

Live connection status

Live preview + detection window

Fixed two-column layout

Clean real-time UI

🧠 Tech Stack

Python

Streamlit

OpenCV

Ultralytics YOLOv8 / YOLO11

PyTorch (CUDA 12.x)

Requests

Recommended Hardware

✔ NVIDIA RTX 3050 or higher

✔ Any CUDA-enabled GPU

▶️ How to Run
🔹 Using Python Virtual Environment
C:\projects\VOLLEYBAL> venv\Scripts\activate
streamlit run dashboard.py

🔹 Directly using Batch File
Run_Dashboard.bat

📄 License

This project is licensed under the MIT License.
You are free to use, modify, distribute, and commercialize this project with attribution.

📜 MIT License (Create LICENSE file)

Create a file named LICENSE in your repo root and paste this exactly:

MIT License

Copyright (c) 2025 Abhinav N Devadiga

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

✅ Final Steps to Push
git add README.md LICENSE
git commit -m "Organized README layout and added MIT license"
git push origin main
