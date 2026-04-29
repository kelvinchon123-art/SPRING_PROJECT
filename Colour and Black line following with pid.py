import time
import cv2
import numpy as np
from picamera2 import Picamera2
from gpiozero import PWMOutputDevice, DigitalOutputDevice

LEFT_BASE  = 0.20  
RIGHT_BASE = 0.20
Kp = 0.015
Kd = 0.001
Ki = 0.0

BLACK_THRESHOLD = 80
STOP_LINE_AREA = 15000

motor_left_speed = PWMOutputDevice(12, frequency=50)
motor_left_in1 = DigitalOutputDevice(5)
motor_left_in2 = DigitalOutputDevice(6)
motor_right_speed = PWMOutputDevice(19, frequency=50)
motor_right_in3 = DigitalOutputDevice(22)
motor_right_in4 = DigitalOutputDevice(26)

def set_motor(left_val, right_val):
    left_val = max(min(left_val, 1.0), -1.0)
    right_val = max(min(right_val, 1.0), -1.0)
    
    if left_val >= 0:
        motor_left_in1.off()
        motor_left_in2.on()
    else:
        motor_left_in1.on()
        motor_left_in2.off()
    motor_left_speed.value = abs(left_val)
    
    if right_val >= 0:
        motor_right_in3.off()
        motor_right_in4.on()
    else:
        motor_right_in3.on()
        motor_right_in4.off()
    motor_right_speed.value = abs(right_val)

def stop():
    set_motor(0, 0)

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "XRGB8888"})
picam2.configure(config)
picam2.start()
time.sleep(2)

last_error = 0
I = 0

current_direction = "UP" 
memory_expiry_time = 0.0

is_on_shortcut = False
shortcut_exit_direction = "UP"

try:
    while True:
        raw_frame = picam2.capture_array("main")
        raw_frame = raw_frame[:, :, :3]
        raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

        if time.time() > memory_expiry_time:
            current_direction = "UP" 

        driver_frame = cv2.resize(raw_frame, (320, 240))
        roi_line = driver_frame[115:175, 40:280]
        
        hsv_roi = cv2.cvtColor(roi_line, cv2.COLOR_BGR2HSV)
        
        lower_yellow = np.array([80, 100, 100])
        upper_yellow = np.array([100, 255, 255])
        mask_yellow = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
        
        lower_red = np.array([100, 100, 100])
        upper_red = np.array([130, 255, 255])
        mask_red = cv2.inRange(hsv_roi, lower_red, upper_red)
        
        color_mask = cv2.bitwise_or(mask_yellow, mask_red)
        color_contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_color_contours = [c for c in color_contours if cv2.contourArea(c) > 150]
        
        gray = cv2.cvtColor(roi_line, cv2.COLOR_BGR2GRAY)
        _, black_mask = cv2.threshold(gray, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_black_contours = [c for c in black_contours if cv2.contourArea(c) > 150] 

        target_contour = None
        
        if len(valid_color_contours) > 0:
            target_contour = max(valid_color_contours, key=cv2.contourArea)
            cv2.imshow("Line Mask", color_mask)
            
            if not is_on_shortcut:
                is_on_shortcut = True
                
                M_color = cv2.moments(target_contour)
                if M_color["m00"] != 0: 
                    entry_cx = int(M_color["m10"] / M_color["m00"]) 
                    
                    if entry_cx > 120:
                        shortcut_exit_direction = "RIGHT" 
                    else:
                        shortcut_exit_direction = "LEFT"
        
        elif len(valid_black_contours) > 0:
            target_contour = max(valid_black_contours, key=cv2.contourArea)
            cv2.imshow("Line Mask", black_mask)
            
            if is_on_shortcut:
                is_on_shortcut = False
                current_direction = shortcut_exit_direction
                memory_expiry_time = time.time() + 3.0 

        if target_contour is not None:
            area = cv2.contourArea(target_contour)
            
            if area > STOP_LINE_AREA:
                stop()
                break 
            
            M = cv2.moments(target_contour)
            if M["m00"] != 0: 
                cx = int(M["m10"] / M["m00"]) 
                
                x, y, w, h = cv2.boundingRect(target_contour)
                
                if w > 100:  
                    if current_direction == "LEFT":
                        cx = x - 10 
                    elif current_direction == "RIGHT":
                        cx = (x + w) - 5
                        
                    elif current_direction == "UP":
                        extTop = tuple(target_contour[target_contour[:, :, 1].argmin()][0])
                        cx = extTop[0]
                
                error = cx - 120
                
                P = error
                I += error
                D = error - last_error
                
                correction = (Kp * P) + (Ki * I) + (Kd * D)
                last_error = error
                
                left_motor_speed = LEFT_BASE + correction
                right_motor_speed = RIGHT_BASE - correction
                set_motor(left_motor_speed, right_motor_speed)
                
                dot_color = (255, 0, 0) if len(valid_color_contours) > 0 else (0, 0, 255)
                cv2.circle(driver_frame, (cx + 40, int(M["m01"] / M["m00"]) + 115), 5, dot_color, -1)

        cv2.imshow("Driver View", driver_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    stop()
    picam2.stop()
    cv2.destroyAllWindows()
