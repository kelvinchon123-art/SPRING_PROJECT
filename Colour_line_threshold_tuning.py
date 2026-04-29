import cv2
import numpy as np
import time
from picamera2 import Picamera2

def nothing(x):
    pass

print("Starting Picamera2 HSV Tuning Dashboard...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "XRGB8888"})
picam2.configure(config)
picam2.start()
time.sleep(2)

cv2.namedWindow('Tuning Dashboard')

cv2.createTrackbar('H Min', 'Tuning Dashboard', 0, 179, nothing)
cv2.createTrackbar('S Min', 'Tuning Dashboard', 0, 255, nothing)
cv2.createTrackbar('V Min', 'Tuning Dashboard', 0, 255, nothing)
cv2.createTrackbar('H Max', 'Tuning Dashboard', 179, 179, nothing)
cv2.createTrackbar('S Max', 'Tuning Dashboard', 255, 255, nothing)
cv2.createTrackbar('V Max', 'Tuning Dashboard', 255, 255, nothing)

try:
    while True:
        frame = picam2.capture_array("main")
        frame = frame[:, :, :3]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos('H Min', 'Tuning Dashboard')
        s_min = cv2.getTrackbarPos('S Min', 'Tuning Dashboard')
        v_min = cv2.getTrackbarPos('V Min', 'Tuning Dashboard')
        
        h_max = cv2.getTrackbarPos('H Max', 'Tuning Dashboard')
        s_max = cv2.getTrackbarPos('S Max', 'Tuning Dashboard')
        v_max = cv2.getTrackbarPos('V Max', 'Tuning Dashboard')

        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])
        
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        color_result = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('Original Camera', frame)
        cv2.imshow('Black & White Mask', mask)
        cv2.imshow('Color Result', color_result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user")
finally:
    picam2.stop()
    cv2.destroyAllWindows()
