# SPRING_PROJECT
# Autonomous 4WD SUV Robot (EEEE1027)

This repository contains the Python software architecture for an autonomous 4WD delivery vehicle built on a Raspberry Pi 4. 

## Core Features:
* **Asynchronous Multiprocessing:** Bypasses the Python Global Interpreter Lock (GIL) to run 30 FPS trajectory control and 5 FPS computer vision concurrently on separate CPU cores.
* **Proportional-Derivative (PD) Kinematics:** Operates at a strict 50 Hz PWM baseline to completely eliminate rotational inertia and mechanical stuttering.
* **Multi-Algorithmic Vision:** Utilizes OpenCV for Inverse Binary tracking, HSV shortcut detection, ORB keypoint extraction, and Hu Moments shape verification. 

## Requirements:
* Python 3.9+
* OpenCV (`cv2`)
* `gpiozero`
