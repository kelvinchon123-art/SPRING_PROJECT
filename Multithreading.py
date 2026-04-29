import time
import cv2
import numpy as np
import threading
import queue
from picamera2 import Picamera2
from gpiozero import PWMOutputDevice, DigitalOutputDevice

def thinker_process(frame_queue, result_queue):
    print("[THINKER] Booting up on a THREAD...")
    
    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    image_files = {
        "Fingerprint": ("/home/user/Desktop/FingerPrint2.png"),
        "Warning": ("/home/user/Desktop/Caution2.png"),
        "Push Button": ("/home/user/Desktop/Button2.png")
    }

    templates = {}
    for name, (filename, threshold) in image_files.items():
        try:
            img = cv2.imread(filename, 0)
            if img is not None:
                kp, des = orb.detectAndCompute(img, None)
                templates[name] = {"image": img, "keypoints": kp, "descriptors": des, "min_matches": threshold}
        except Exception:
            pass

    try:
        template_img = cv2.imread('/home/user/Desktop/Recycle.png', 0)
        if template_img is not None:
            _, temp_thresh = cv2.threshold(template_img, 127, 255, cv2.THRESH_BINARY_INV)
            temp_contours, _ = cv2.findContours(temp_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            temp_contours = sorted(temp_contours, key=cv2.contourArea, reverse=True)
            template_arrow = temp_contours[0]
    except Exception:
        pass

    print("[THINKER] Ready and waiting for frames!")

    while True:
        frame = frame_queue.get()
        if frame is None:
            break
            
        detected_symbols = []
        special_logo_rects = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, gray_mask = cv2.threshold(blurred_gray, 130, 255, cv2.THRESH_BINARY_INV)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(hsv)
        blurred_s = cv2.GaussianBlur(s, (5, 5), 0)
        _, sat_mask = cv2.threshold(blurred_s, 100, 255, cv2.THRESH_BINARY)

        kp_frame, des_frame = orb.detectAndCompute(gray, None)
        if des_frame is not None:
            for name, data in templates.items():
                matches = bf.match(data["descriptors"], des_frame)
                if len(matches) > data["min_matches"]:
                    src_pts = np.float32([data["keypoints"][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if M is not None:
                        h_img, w_img = data["image"].shape
                        pts = np.float32([[0, 0], [0, h_img - 1], [w_img - 1, h_img - 1], [w_img - 1, 0]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)
                        
                        if cv2.isContourConvex(np.int32(dst)):
                            bx, by, bw, bh = cv2.boundingRect(np.int32(dst))
                            if (bw * bh) > 1500:
                                special_logo_rects.append((bx, by, bw, bh))
                                detected_symbols.append(name)

        contours_qr, hierarchy = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        found_markers = []
        if hierarchy is not None:
            for i, contour in enumerate(contours_qr):
                if cv2.contourArea(contour) > 500:
                    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
                    if len(approx) == 4 and hierarchy[0][i][2] != -1:
                        child_approx = cv2.approxPolyDP(contours_qr[hierarchy[0][i][2]], 0.04 * cv2.arcLength(contours_qr[hierarchy[0][i][2]], True), True)
                        if len(child_approx) == 4: found_markers.append(approx)

        if len(found_markers) >= 3:
            x, y, w, h = cv2.boundingRect(np.vstack(found_markers))
            special_logo_rects.append((x, y, w, h))
            detected_symbols.append("QR Code")

        contours_recy, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found_arrows = []
        try:
            for contour in contours_recy:
                if cv2.contourArea(contour) < 500: continue
                match_score = cv2.matchShapes(template_arrow, contour, cv2.CONTOURS_MATCH_I1, 0.0)

                if match_score < 0.5:
                    found_arrows.append(contour)
                    
            if len(found_arrows) >= 3:
                x, y, w, h = cv2.boundingRect(np.vstack(found_arrows))
                special_logo_rects.append((x, y, w, h))
                detected_symbols.append("Recycle Logo")
        except NameError:
            pass

        contours_shapes, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_shapes:
            area = cv2.contourArea(contour)
            if area < 800: continue

            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = int(x + (w/2)), int(y + (h/2))

            if any(sx < cx < (sx + sw) and sy < cy < (sy + sh) for (sx, sy, sw, sh) in special_logo_rects):
                continue 

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
            vertices = len(approx)
            solidity = float(area) / cv2.contourArea(cv2.convexHull(contour)) if cv2.contourArea(cv2.convexHull(contour)) > 0 else 0
            shape_name = "Unknown"

            if vertices in [7, 8, 9, 10] and 0.55 < solidity < 0.65: 
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    moment_cx, moment_cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
                    extLeft, extRight = tuple(contour[contour[:, :, 0].argmin()][0]), tuple(contour[contour[:, :, 0].argmax()][0])
                    extTop, extBot = tuple(contour[contour[:, :, 1].argmin()][0]), tuple(contour[contour[:, :, 1].argmax()][0])
                    if w > h: shape_name = "Arrow: LEFT" if abs(extRight[0] - moment_cx) > abs(moment_cx - extLeft[0]) else "Arrow: RIGHT"
                    else: shape_name = "Arrow: DOWN" if abs(moment_cy - extTop[1]) > abs(extBot[1] - moment_cy) else "Arrow: UP"

            if shape_name != "Unknown":
                detected_symbols.append(shape_name)

        if len(detected_symbols) > 0:
            unique_symbols = list(set(detected_symbols))
            result_queue.put(unique_symbols)


if __name__ == '__main__':
    frame_queue = queue.Queue(maxsize=1) 
    result_queue = queue.Queue()
    
    thinker = threading.Thread(target=thinker_process, args=(frame_queue, result_queue))
    thinker.daemon = True 
    thinker.start()

    LEFT_BASE  = 0.20
    RIGHT_BASE = 0.20
    Kp = 0.009
    Kd = 0.006
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
            motor_left_in1.off(); motor_left_in2.on()
        else:
            motor_left_in1.on(); motor_left_in2.off()
        motor_left_speed.value = abs(left_val)
        
        if right_val >= 0:
            motor_right_in3.off(); motor_right_in4.on()
        else:
            motor_right_in3.on(); motor_right_in4.off()
        motor_right_speed.value = abs(right_val)

    def stop():
        set_motor(0, 0)

    print("Starting Raspberry Pi Camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "XRGB8888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    
    print("=== PID + MULTITHREADING VISION BOOTING ===")
    last_error = 0
    I = 0
    
    ignore_vision_until = 0.0
    current_direction = "UP" 
    arrow_expiry_time = 0.0
    is_on_shortcut = False
    shortcut_exit_direction = "UP"

    try:
        while True:
            raw_frame = picam2.capture_array()
            raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            
            if time.time() > ignore_vision_until:
                if not frame_queue.full():
                    thinker_roi = raw_frame[0:350, 0:640]
                    try:
                        frame_queue.put_nowait(thinker_roi) 
                    except queue.Full:
                        pass 
            else:
                while not result_queue.empty():
                    result_queue.get()

            if time.time() > arrow_expiry_time:
                current_direction = "UP" 

            if time.time() > ignore_vision_until:
                while not result_queue.empty():
                    found_symbols = result_queue.get()
                    action_taken = False
                    
                    for sym in found_symbols:
                        print(f"\n[THINKER SAYS]: I see a {sym}!!")
                        
                        if sym in ["Push Button", "Warning"]:
                            stop()
                            time.sleep(3.0)
                            action_taken = True
                            break
                            
                        elif sym == "Recycle Logo":
                            set_motor(0.7, -0.7) 
                            time.sleep(2.8)      
                            stop()
                            action_taken = True
                            break
                            
                        elif "Arrow" in sym:
                            direction = sym.split(": ")[1]
                            current_direction = direction
                            arrow_expiry_time = time.time() + 2.0 
                            break
                            
                        elif sym in ["Fingerprint","QR Code"]:
                            action_taken = True
                            break
                            
                    if action_taken:
                        ignore_vision_until = time.time() + 2.0
                        while not result_queue.empty(): result_queue.get()
                        last_error = 0
                        I = 0
                        break 
            
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
            
            gray = cv2.cvtColor(roi_line, cv2.COLOR_RGB2GRAY)
            _, black_mask = cv2.threshold(gray, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
            black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_black_contours = [c for c in black_contours if cv2.contourArea(c) > 150] 

            target_contour = None
            
            if len(valid_color_contours) > 0 and time.time() > arrow_expiry_time:
                target_contour = max(valid_color_contours, key=cv2.contourArea)
                cv2.imshow("Line Follower ROI", color_mask)
                
                if not is_on_shortcut:
                    is_on_shortcut = True
                    
                    M_color = cv2.moments(target_contour)
                    if M_color["m00"] != 0: 
                        entry_cx = int(M_color["m10"] / M_color["m00"]) 
                        
                        if entry_cx > 100:
                            shortcut_exit_direction = "RIGHT" 
                        else:
                            shortcut_exit_direction = "LEFT"
            
            elif len(valid_black_contours) > 0:
                target_contour = max(valid_black_contours, key=cv2.contourArea)
                cv2.imshow("Line Follower ROI", black_mask)
                
                if is_on_shortcut:
                    is_on_shortcut = False
                    current_direction = shortcut_exit_direction
                    arrow_expiry_time = time.time() + 3.0 

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
                            cx = x -10 
                        elif current_direction == "RIGHT":
                            cx = (x + w) -5
                            
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

            cv2.imshow("Driver Full View (Shrunk to 320p)", driver_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nUser Interrupted!")
    finally:
        stop()
        picam2.stop()
        frame_queue.put(None) 
        thinker.join(timeout=1.0) 
        cv2.destroyAllWindows()
        print("Shutdown Complete.")
