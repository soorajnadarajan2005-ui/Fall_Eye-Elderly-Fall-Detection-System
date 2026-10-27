from flask import Flask, redirect, render_template, request, url_for, Response, abort
import numpy as np
from collections import deque
import time

# try import cv2; allow graceful fallback if not installed
try:
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False


app = Flask(__name__)

# Fall detection globals
fall_detected = False
fall_alert_time = 0
fall_confirmed_count = 0  # Frames with fall indicators
no_fall_count = 0  # Consecutive frames without fall
last_positions = deque(maxlen=30)  # Store last 30 positions for motion analysis
last_heights = deque(maxlen=30)  # Store person heights
last_widths = deque(maxlen=30)  # Store person widths
frame_count = 0

@app.route('/hello/<name>')
def hello_name(name):
    return 'Hello %s ! Welcome to our Application.' % name

@app.route('/index')
def index():
    return render_template('index.html')
    
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/logout')
def logout():
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html', has_cv2=HAS_CV2)


def detect_fall_function(frame):
    """
    Simple yet effective fall detection using OpenCV:
    1. Skin color detection to find person
    2. Aspect ratio analysis
    3. Position tracking
    
    Returns: (fall_detected, annotated_frame, confidence)
    """
    global fall_detected, fall_alert_time, fall_confirmed_count, no_fall_count
    
    if not HAS_CV2:
        return False, frame, 0
    
    h, w = frame.shape[:2]
    fall_detected_current = False
    confidence = 0
    status_text = "NORMAL"
    status_color = (0, 255, 0)
    person_detected = False
    
    try:
        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range in HSV
        lower_skin1 = np.array([0, 15, 60], dtype=np.uint8)
        upper_skin1 = np.array([17, 170, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        
        lower_skin2 = np.array([160, 15, 60], dtype=np.uint8)
        upper_skin2 = np.array([180, 170, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours and len(contours) > 0:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Minimum area threshold (person must be substantial)
            if area > 1500:
                person_detected = True
                x, y, box_w, box_h = cv2.boundingRect(largest_contour)
                
                # Ensure valid dimensions
                if box_w > 0 and box_h > 0:
                    # Store measurements
                    last_heights.append(box_h)
                    last_widths.append(box_w)
                    
                    center_y = (y + box_h / 2) / h
                    last_positions.append(center_y)
                    
                    # Calculate aspect ratio (height / width)
                    aspect_ratio = box_h / box_w
                    
                    # --- Fall Detection Logic ---
                    fall_score = 0
                    max_score = 0
                    
                    # Score 1: Low aspect ratio (horizontal = fallen)
                    max_score += 2
                    if aspect_ratio < 0.7:  # Less than 70% = horizontal
                        fall_score += 2
                    elif aspect_ratio < 0.85:
                        fall_score += 1
                    
                    # Score 2: Low position in frame
                    max_score += 2
                    if center_y > 0.65:  # Bottom 35%
                        fall_score += 2
                    elif center_y > 0.55:  # Bottom 45%
                        fall_score += 1
                    
                    # Score 3: Height reduction over time
                    max_score += 1
                    if len(last_heights) >= 10:
                        recent_heights = list(last_heights)[-10:]
                        avg_height = np.mean(recent_heights)
                        if avg_height > 0 and (avg_height - box_h) / avg_height > 0.30:
                            fall_score += 1
                    
                    # Score 4: Width increase (spreading)
                    max_score += 1
                    if len(last_widths) >= 10:
                        recent_widths = list(last_widths)[-10:]
                        avg_width = np.mean(recent_widths)
                        if avg_width > 0 and (box_w - avg_width) / avg_width > 0.20:
                            fall_score += 1
                    
                    # Calculate confidence (0-1)
                    confidence = fall_score / max_score
                    
                    # Alert Logic
                    if confidence >= 0.60:
                        fall_confirmed_count += 1
                        no_fall_count = 0
                        
                        if fall_confirmed_count >= 6:  # 6 frames at 24fps ~ 250ms
                            fall_detected_current = True
                            fall_detected = True
                            fall_alert_time = time.time()
                            status_text = f"FALL DETECTED! ({int(confidence*100)}%)"
                            status_color = (0, 0, 255)
                        else:
                            status_text = f"SUSPICIOUS ({int(confidence*100)}%)"
                            status_color = (0, 165, 255)
                    else:
                        fall_confirmed_count = max(0, fall_confirmed_count - 1)
                        no_fall_count += 1
                        if no_fall_count > 5:
                            fall_confirmed_count = 0
                        status_text = "NORMAL"
                        status_color = (0, 255, 0)
                    
                    # Draw bounding box
                    box_color = (0, 0, 255) if fall_detected_current else ((0, 165, 255) if fall_confirmed_count > 1 else (0, 255, 0))
                    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), box_color, 2)
                    
                    # Draw center point
                    center_x = x + box_w // 2
                    center_y_px = y + box_h // 2
                    cv2.circle(frame, (center_x, center_y_px), 5, box_color, -1)
                    
                    # Display info
                    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                    cv2.putText(frame, f"Aspect: {aspect_ratio:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, f"Conf: {fall_confirmed_count}/6", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, f"Score: {confidence:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if not person_detected:
            cv2.putText(frame, "NO PERSON DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            fall_confirmed_count = 0
            no_fall_count += 1
    
    except Exception as e:
        print(f"Detection error: {e}")
        cv2.putText(frame, "DETECTION ERROR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return fall_detected_current, frame, confidence
    """
    Advanced fall detection using multiple methods:
    1. Primary: MediaPipe pose detection (body landmarks)
    2. Fallback: Simple motion/color detection
    
    Returns: (fall_detected, annotated_frame, confidence)
    """
    global fall_detected, fall_alert_time, fall_confirmed_count, no_fall_count
    
    if not HAS_CV2:
        return False, frame, 0
    
    h, w = frame.shape[:2]
    fall_detected_current = False
    confidence = 0
    status_text = "NORMAL"
    status_color = (0, 255, 0)
    person_detected = False
    
    # Try MediaPipe detection first
    if pose_detector and HAS_MEDIAPIPE:
        try:
            # Convert frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(frame_rgb)
            
            if results.pose_landmarks:
                person_detected = True
                # Extract key landmarks
                landmarks = results.pose_landmarks.landmark
                
                # Get important body parts (in image coordinates)
                nose = landmarks[0]
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]
                left_hip = landmarks[23]
                right_hip = landmarks[24]
                left_ankle = landmarks[27]
                right_ankle = landmarks[28]
                
                # Filter out low confidence detections
                min_confidence = 0.4
                if nose.visibility < min_confidence:
                    no_fall_count += 1
                    if no_fall_count > 10:
                        fall_confirmed_count = 0
                    return False, frame, 0
                
                # Calculate bounding box from landmarks
                x_coords = [l.x for l in [nose, left_shoulder, right_shoulder, left_hip, left_ankle, right_ankle] if l.visibility > min_confidence]
                y_coords = [l.y for l in [nose, left_shoulder, right_shoulder, left_hip, left_ankle, right_ankle] if l.visibility > min_confidence]
                
                if x_coords and y_coords:
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    x_min_px, x_max_px = int(x_min * w), int(x_max * w)
                    y_min_px, y_max_px = int(y_min * h), int(y_max * h)
                    
                    bbox_width = max(1, x_max_px - x_min_px)
                    bbox_height = max(1, y_max_px - y_min_px)
                    
                    # Store measurements
                    last_heights.append(bbox_height)
                    last_widths.append(bbox_width)
                    
                    center_y = (y_min_px + y_max_px) / 2 / h
                    last_positions.append(center_y)
                    
                    # --- Fall Detection Analysis ---
                    fall_indicators = 0
                    total_criteria = 0
                    
                    # Criterion 1: Body is horizontal (lying down)
                    total_criteria += 1
                    body_height_ratio = bbox_height / bbox_width
                    if body_height_ratio < 0.7:  # More horizontal than vertical
                        fall_indicators += 1
                    
                    # Criterion 2: Person center is low in frame
                    total_criteria += 1
                    if center_y > 0.55:  # Lower 45% of frame
                        fall_indicators += 1
                    
                    # Criterion 3: Ankle higher than hip (legs up)
                    total_criteria += 1
                    avg_hip_y = (left_hip.y + right_hip.y) / 2
                    avg_ankle_y = (left_ankle.y + right_ankle.y) / 2
                    if avg_ankle_y < avg_hip_y - 0.05:
                        fall_indicators += 1
                    
                    # Criterion 4: Shoulder drop
                    total_criteria += 1
                    avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
                    if avg_shoulder_y > avg_hip_y - 0.15:
                        fall_indicators += 1
                    
                    # Criterion 5: Height reduction
                    total_criteria += 1
                    if len(last_heights) >= 10:
                        recent_heights = list(last_heights)[-10:]
                        avg_height = np.mean(recent_heights)
                        if avg_height > 0 and (avg_height - bbox_height) / avg_height > 0.25:
                            fall_indicators += 1
                    
                    confidence = fall_indicators / total_criteria
                    
                    # Alert Logic
                    if confidence >= 0.55:
                        fall_confirmed_count += 1
                        no_fall_count = 0
                        
                        if fall_confirmed_count >= 8:
                            fall_detected_current = True
                            fall_detected = True
                            fall_alert_time = time.time()
                            status_text = f"FALL DETECTED! ({int(confidence*100)}%)"
                            status_color = (0, 0, 255)
                        else:
                            status_text = f"SUSPICIOUS ({int(confidence*100)}%)"
                            status_color = (0, 165, 255)
                    else:
                        fall_confirmed_count = max(0, fall_confirmed_count - 1)
                        no_fall_count += 1
                        if no_fall_count > 5:
                            fall_confirmed_count = 0
                    
                    # Draw bounding box
                    box_color = (0, 0, 255) if fall_detected_current else ((0, 165, 255) if fall_confirmed_count > 2 else (0, 255, 0))
                    cv2.rectangle(frame, (x_min_px, y_min_px), (x_max_px, y_max_px), box_color, 2)
                    
                    # Draw landmarks
                    for landmark in [nose, left_shoulder, right_shoulder, left_hip, left_ankle, right_ankle]:
                        if landmark.visibility > min_confidence:
                            x, y = int(landmark.x * w), int(landmark.y * h)
                            cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)
                    
                    # Display info
                    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                    cv2.putText(frame, f"Conf: {fall_confirmed_count}/8", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, f"Score: {confidence:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
        except Exception as e:
            print(f"MediaPipe error: {e}")
            person_detected = False
    
    # Fallback: Simple detection using contours if MediaPipe fails
    if not person_detected:
        try:
            # Convert to HSV for skin detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Skin color range
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Also detect by color (simple approach)
            lower_skin2 = np.array([160, 20, 70], dtype=np.uint8)
            upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
            mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            mask = cv2.bitwise_or(mask, mask2)
            
            # Apply morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour (person)
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 500:  # Minimum area for person
                    person_detected = True
                    x, y, rw, rh = cv2.boundingRect(largest_contour)
                    
                    # Store measurements
                    last_heights.append(rh)
                    last_widths.append(rw)
                    center_y = (y + rh / 2) / h
                    last_positions.append(center_y)
                    
                    # Simple fall detection: person is low and wide
                    aspect_ratio = rh / (rw + 1)
                    
                    if aspect_ratio < 0.8 and center_y > 0.55:
                        fall_confirmed_count += 1
                        confidence = min(0.8, aspect_ratio * 0.5 + (0.6 if center_y > 0.6 else 0))
                        status_text = f"POSSIBLE FALL ({int(confidence*100)}%)"
                        status_color = (0, 165, 255)
                    else:
                        fall_confirmed_count = max(0, fall_confirmed_count - 1)
                        status_text = "NORMAL"
                        status_color = (0, 255, 0)
                    
                    cv2.rectangle(frame, (x, y), (x + rw, y + rh), status_color, 2)
                    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                    cv2.putText(frame, f"Aspect: {aspect_ratio:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            if not person_detected:
                cv2.putText(frame, "NO PERSON DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                fall_confirmed_count = 0
                no_fall_count += 1
        
        except Exception as e:
            print(f"Fallback detection error: {e}")
            cv2.putText(frame, "DETECTION ERROR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return fall_detected_current, frame, confidence


def gen_frames():
    global fall_detected, fall_alert_time
    
    if not HAS_CV2:
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return
    
    frame_count = 0
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame")
                break
            
            frame_count += 1
            
            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 480))
            
            # Detect falls
            try:
                is_fall, annotated_frame, confidence = detect_fall(frame)
            except Exception as e:
                print(f"Error in fall detection: {e}")
                annotated_frame = frame
                cv2.putText(annotated_frame, "Live Camera Feed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                print(f"Failed to encode frame {frame_count}")
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        print(f"Error in gen_frames: {e}")
    finally:
        cap.release()
        print("Camera released")


@app.route('/fall_status')
def fall_status():
    """Return current fall detection status as JSON"""
    global fall_detected, fall_alert_time
    
    # Reset fall alert after 5 seconds
    if fall_detected and time.time() - fall_alert_time > 5:
        fall_detected = False
    
    return {'fall_detected': fall_detected}


@app.route('/video_feed')
def video_feed():
    if not HAS_CV2:
        abort(501, description='OpenCV (cv2) not available on server')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/loginproc', methods=['POST', 'GET'])
def loginproc():
    if request.method  == 'POST':
        user = request.form['username']
        password = request.form['password']
        if user == 'admin' and password == 'admin':
            return redirect(url_for('home'))
        else:
            return redirect(url_for('login'))
    else:
        user=request.args.get('username')
        return redirect(url_for('success', username=user))

@app.route('/success/<username>')
def success(username):
    return 'welcome %s' % username


if __name__ == '__main__':
    app.run(debug=True)
