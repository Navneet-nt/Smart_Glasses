# Smart Obstacle Detection Assistant for Smart Glasses

This project implements a **Smart Obstacle Detection Assistant** designed for smart glasses. The system uses real-time computer vision and voice interactions to detect obstacles, provide audible warnings, and offer navigation instructions for safer mobility.

## Features

- **Real-Time Obstacle Detection:**  
  Utilizes a YOLO-based model (`obstacle_detection.pt`) to detect obstacles in video streams.
  
- **Obstacle Classification:**  
  Differentiates between *unavoidable obstacles* (e.g., barriers, gates, roadblocks) and *avoidable obstacles* (e.g., cars, doors, drains, fences, poles, trees, rickshaws, potholes).

- **Audible Warnings and Navigation Instructions:**  
  Uses a text-to-speech engine (pyttsx3) to announce obstacle locations and navigation instructions based on detected objects.

- **Voice Command Interaction:**  
  Implements speech recognition to allow voice commands such as:
  - **Stop:** End the detection process.
  - **Pause/Resume:** Control the detection system.
  - **Quiet/Verbose:** Toggle the verbosity of announcements.
  - **Help:** List available commands.
  - **Surroundings Announcement:** Request an overview of nearby obstacles.

- **Multi-threaded Operation:**  
  Uses threading and queues to handle video processing, voice command listening, and text-to-speech concurrently.

## Requirements

- **Python 3.x**

- **Python Packages:**
  - `ultralytics` (for YOLO object detection)
  - `opencv-python`
  - `pyttsx3`
  - `speech_recognition`
  - `numpy`
  - Other standard libraries: `threading`, `queue`, `time`, etc.

Install the required packages using pip:

```bash
pip install ultralytics opencv-python pyttsx3 SpeechRecognition numpy
