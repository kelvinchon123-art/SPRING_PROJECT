import cv2
import numpy as np
import time
from picamera2 import Picamera2

# ==========================================
# 1. SETUP: ORB (For Fingerprint, Warning, Button)
# ==========================================
orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Note: Using your specific Raspberry Pi folder paths!
image_files = {
    "Fingerprint": "/home/user/FingerPrint.png",
    "Warning": "/home/user/Caution.png",
    "Push Button": "/home/user/Button.png"
}

templates = {}
print("Loading ORB Templates...")
for name, filename in image_files.items():
    try:
        img = cv2.imread(filename, 0)
        if img is not None:
            kp, des = orb.detectAndCompute(img, None)
            templates[name] = {"image": img, "keypoints": kp, "descriptors": des}
            print(f"[*] Loaded {name} successfully.")
        else:
            print(f"[!] Could not find {filename}. Is it in the correct folder?")
    except Exception as e:
        print(f"[!] Error loading {name}: {e}")

# ==========================================
# 2. SETUP: RECYCLE ARROW TEMPLATE
# ==========================================
try:
    template_img = cv2.imread('/home/user/Recycle.png', 0)
    if template_img is not None:
        _, temp_thresh = cv2.threshold(template_img, 127, 255, cv2.THRESH_BINARY_INV)
        temp_contours, _ = cv2.findContours(temp_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        temp_contours = sorted(temp_contours, key=cv2.contourArea, reverse=True)
        template_arrow = temp_contours[0] 
    else:
        print("[!] Could not find Recycle.png.")
except Exception as e:
    print(f"[!] Error loading Recycle template: {e}")

# ==========================================
# 3. START WEBCAM (THE RASPBERRY PI WAY)
# ==========================================
print("Starting Raspberry Pi Camera...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "XRGB8888"})
picam2.configure(config)
picam2.start()
time.sleep(2) # Warm up camera

MIN_MATCH_COUNT = 15

while True:
    # Grab the frame directly from the Pi's memory
    frame = picam2.capture_array("main")
    frame = frame[:, :, :3] # Clean the alpha channel
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Convert to OpenCV format

    # --- THE "DO NOT DISTURB" ZONES ---
    # We will store the (x, y, w, h) of every special logo here
    special_logo_rects = []

    # --- PRE-PROCESSING A: THE GRAYSCALE MASK (For ORB, QR, and Recycle) ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, gray_mask = cv2.threshold(blurred_gray, 130, 255, cv2.THRESH_BINARY_INV)

    # --- PRE-PROCESSING B: THE SATURATION MASK (For Colored Shapes & Arrows) ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    blurred_s = cv2.GaussianBlur(s, (5, 5), 0)
    _, sat_mask = cv2.threshold(blurred_s, 100, 255, cv2.THRESH_BINARY)


    # ==========================================
    # BRAIN 1: ORB MATCHING (With Overlap Suppression!)
    # ==========================================
    kp_frame, des_frame = orb.detectAndCompute(gray, None)
    
    # Temporary list to hold all ORB guesses before we draw them
    orb_candidates = []

    if des_frame is not None:
        for name, data in templates.items():
            matches = bf.match(data["descriptors"], des_frame)
            
            if len(matches) > MIN_MATCH_COUNT:
                src_pts = np.float32([data["keypoints"][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is not None:
                    h_img, w_img = data["image"].shape
                    pts = np.float32([[0, 0], [0, h_img - 1], [w_img - 1, h_img - 1], [w_img - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    bx, by, bw, bh = cv2.boundingRect(np.int32(dst))
                    
                    # Save the guess to our list instead of drawing it immediately!
                    orb_candidates.append({
                        "name": name,
                        "matches": len(matches),
                        "polygon": dst,
                        "rect": (bx, by, bw, bh)
                    })

    # --- NON-MAXIMUM SUPPRESSION LOGIC ---
    # Sort all our guesses by the number of matches (Highest score goes first!)
    orb_candidates.sort(key=lambda x: x["matches"], reverse=True)
    
    # We keep track of the boxes we've officially drawn
    drawn_orb_boxes = []

    for candidate in orb_candidates:
        bx, by, bw, bh = candidate["rect"]
        cx, cy = bx + (bw / 2), by + (bh / 2)
        
        # Check if this shape is trapped inside a higher-scoring shape
        is_overlapping = False
        for (dx, dy, dw, dh) in drawn_orb_boxes:
            if dx < cx < (dx + dw) and dy < cy < (dy + dh):
                is_overlapping = True
                break
                
        if not is_overlapping:
            # It passed the test! Draw the box!
            dst = candidate["polygon"]
            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"{candidate['name']} ({candidate['matches']})", 
                        (int(dst[0][0][0]), int(dst[0][0][1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Add it to our drawn boxes so weaker matches can't overlap it
            drawn_orb_boxes.append(candidate["rect"])
            
            # --- ADD TO DO-NOT-DISTURB FOR BRAIN 4 ---
            special_logo_rects.append(candidate["rect"])

    # ==========================================
    # BRAIN 2: QR MARKER (Hierarchical Contours)
    # ==========================================
    contours_qr, hierarchy = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    found_markers = []

    if hierarchy is not None:
        for i, contour in enumerate(contours_qr):
            if cv2.contourArea(contour) < 500: continue 
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

            if len(approx) == 4:
                child_idx = hierarchy[0][i][2]
                if child_idx != -1: 
                    child_approx = cv2.approxPolyDP(contours_qr[child_idx], 0.04 * cv2.arcLength(contours_qr[child_idx], True), True)
                    if len(child_approx) == 4:
                        found_markers.append(approx)
                        cv2.drawContours(frame, [approx], 0, (255, 0, 0), 2)

    if len(found_markers) >= 3:
        x, y, w, h = cv2.boundingRect(np.vstack(found_markers))
        
        # --- ADD TO DO-NOT-DISTURB ---
        special_logo_rects.append((x, y, w, h))
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(frame, "QR DETECTED!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)


    # ==========================================
    # BRAIN 3: RECYCLE LOGO (Hu Moments)
    # ==========================================
    contours_recy, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found_arrows = []

    try:
        for contour in contours_recy:
            if cv2.contourArea(contour) < 500: continue
            match_score = cv2.matchShapes(template_arrow, contour, cv2.CONTOURS_MATCH_I1, 0.0)
            
            if match_score < 0.65:
                found_arrows.append(contour)

        if len(found_arrows) >= 3:
            x, y, w, h = cv2.boundingRect(np.vstack(found_arrows))
            
            # --- ADD TO DO-NOT-DISTURB ---
            special_logo_rects.append((x, y, w, h))
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, "RECYCLE LOGO!", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
    except NameError:
        pass


    # ==========================================
    # BRAIN 4: COLORED SHAPES & ARROWS
    # ==========================================
    master_shape_mask = cv2.bitwise_or(sat_mask, gray_mask)
    contours_shapes, _ = cv2.findContours(master_shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours_shapes:
        area = cv2.contourArea(contour)
        if area < 800: continue

        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = int(x + (w/2)), int(y + (h/2))

        # --- THE SPATIAL MASKING CHECK ---
        # Is the center of this shape inside ANY of the special logo boxes?
        is_inside_special = False
        for (sx, sy, sw, sh) in special_logo_rects:
            if sx < cx < (sx + sw) and sy < cy < (sy + sh):
                is_inside_special = True
                break # It's inside a special logo! Stop checking.
        
        if is_inside_special:
            continue # ABORT! Skip this shape entirely so we don't double-draw!

        # (If it survived the check, do the math!)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        vertices = len(approx)
        
        aspect_ratio = float(w) / h
        extent = float(area) / (w * h)
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        shape_name = "Unknown"

        if vertices == 3: shape_name = "Triangle"
        elif vertices == 4:
            if 0.45 <= extent <= 0.55: shape_name = "Diamond"
            elif extent > 0.85: shape_name = "Square" if 0.90 <= aspect_ratio <= 1.10 else "Rectangle"
            else: shape_name = "Trapezoid"
        elif vertices == 5: shape_name = "Pentagon"
        elif vertices == 6: shape_name = "Hexagon"
        elif vertices == 8: shape_name = "Octagon"
        elif vertices == 12 and 0.85 < solidity < 0.95: shape_name = "Cross"
        elif vertices in [7, 8, 9, 10] and 0.55 < solidity < 0.65: 
            M = cv2.moments(contour)
            if M['m00'] != 0:
                moment_cx, moment_cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
                extLeft, extRight = tuple(contour[contour[:, :, 0].argmin()][0]), tuple(contour[contour[:, :, 0].argmax()][0])
                extTop, extBot = tuple(contour[contour[:, :, 1].argmin()][0]), tuple(contour[contour[:, :, 1].argmax()][0])

                if w > h:
                    shape_name = "Arrow: LEFT" if abs(extRight[0] - moment_cx) > abs(moment_cx - extLeft[0]) else "Arrow: RIGHT"
                else:
                    shape_name = "Arrow: DOWN" if abs(moment_cy - extTop[1]) > abs(extBot[1] - moment_cy) else "Arrow: UP"
        elif vertices == 10 and solidity < 0.55: shape_name = "Star"
        elif vertices > 6:
            if solidity > 0.90: shape_name = "Half Circle"
            elif 0.70 <= solidity <= 0.85: shape_name = "Partial Circle"

        # ==========================================
        # 5. DRAW THE RESULTS & X-RAY DEBUGGER
        # ==========================================
        DEBUG_MODE = True 

        if shape_name != "Unknown":
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)
            cv2.putText(frame, shape_name, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if DEBUG_MODE:
            debug_text = f"V:{vertices} S:{solidity:.2f} AR:{aspect_ratio:.2f}"
            cv2.putText(frame, debug_text, (x, y + h + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # ==========================================
    # SHOW EVERYTHING
    # ==========================================
    cv2.imshow("Master Symbol Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

# Clean up cleanly
picam2.stop()
cv2.destroyAllWindows()
