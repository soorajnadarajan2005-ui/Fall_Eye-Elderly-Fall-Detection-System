from flask import Flask, redirect, render_template, request, url_for, Response, abort, jsonify
import numpy as np
from collections import deque
import time
import json
import os
from datetime import datetime, timedelta
import smtplib
import csv
import io
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

# Optional pose estimation with MediaPipe for improved fall detection
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose_detector = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
except (ImportError, Exception):
    mp = None
    mp_pose = None
    mp_drawing = None
    pose_detector = None
    HAS_MEDIAPIPE = False

app = Flask(__name__)

# Email configuration
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SENDER_EMAIL = os.environ.get('GMAIL_SENDER', '').strip()
SENDER_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD', '').strip()
RECIPIENT_EMAIL = os.environ.get('FALL_ALERT_RECIPIENT', '').strip()

# Global variables
fall_alert = False
alert_time = 0
potential_fall_alert = False
potential_fall_time = 0
fall_count = 0
no_fall_count = 0
fall_event_active = False
confidence_history = deque(maxlen=25)
position_history = deque(maxlen=40)
height_history = deque(maxlen=40)
width_history = deque(maxlen=40)
fall_transition_history = deque(maxlen=15)
upright_history = deque(maxlen=15)
previous_frame = None
frame_diff_history = deque(maxlen=15)
email_sent_for_current_alert = False
potential_email_attempted = False

# Runtime settings (dynamic)
# Sensitivity control removed for consistent professional operation.
FALL_CONFIDENCE_THRESHOLD = 0.60  # Lowered from 0.70 for better detection
POTENTIAL_FALL_THRESHOLD = 0.45  # Alert threshold lowered from 50%
email_alert_enabled = True

# Fall data storage
fall_data_file = 'fall_records.json'
fall_records = []


def send_fall_email(subject, body):
    """Send an alert through Gmail using credentials supplied in the environment."""
    if not all((SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL)):
        print("[ERROR] Fall email not sent: configure GMAIL_SENDER, GMAIL_APP_PASSWORD, and FALL_ALERT_RECIPIENT.")
        return False

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
    except (smtplib.SMTPException, OSError) as exc:
        print(f"[ERROR] Failed to send fall alert email: {exc}")
        return False

    print(f"[EMAIL] Fall alert sent to {RECIPIENT_EMAIL}")
    return True


def normalize_fall_record_ids():
    """Ensure every fall record has a unique integer ID."""
    global fall_records
    current_max = max((r.get('id', 0) for r in fall_records), default=0)
    for record in fall_records:
        if 'id' not in record or not isinstance(record.get('id'), int):
            current_max += 1
            record['id'] = current_max


def load_fall_records():
    """Load fall records from file"""
    global fall_records
    if os.path.exists(fall_data_file):
        try:
            with open(fall_data_file, 'r') as f:
                fall_records = json.load(f)
            normalize_fall_record_ids()
            save_fall_records()
        except:
            fall_records = []
    return fall_records


def save_fall_records():
    """Save fall records to file"""
    with open(fall_data_file, 'w') as f:
        json.dump(fall_records, f, indent=2)

def get_next_record_id():
    if not fall_records:
        return 1
    return max((r.get('id', 0) for r in fall_records), default=0) + 1


def add_fall_record(confidence, x, y, width, height, frame_data=None):
    """Add a fall detection record"""
    global fall_records
    record = {
        'id': get_next_record_id(),
        'timestamp': datetime.now().isoformat(),
        'confidence': float(confidence),
        'location': {'x': int(x), 'y': int(y), 'width': int(width), 'height': int(height)},
        'severity': 'HIGH' if confidence > 0.75 else 'MEDIUM' if confidence > 0.50 else 'LOW'
    }
    fall_records.append(record)
    save_fall_records()
    return record

# Load initial records
load_fall_records()

# Global background subtractor for motion detection
bg_subtractor = None
if HAS_CV2:
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

if HAS_CV2:
    KERNEL_LARGE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    KERNEL_SMALL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    KERNEL_MEDIUM = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
else:
    KERNEL_LARGE = KERNEL_SMALL = KERNEL_MEDIUM = None


def detect_motion_contours(frame, prev_frame):
    """Detect person using contours from frame difference and background subtraction"""
    if not HAS_CV2 or prev_frame is None or bg_subtractor is None:
        return None
    
    try:
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)
        
        # Morphological operations to clean up
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, KERNEL_SMALL)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, KERNEL_MEDIUM)
        fg_mask = cv2.dilate(fg_mask, KERNEL_LARGE, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by area (person should have reasonable area)
            if area > 2000 and area < 100000:
                x, y, w, h = cv2.boundingRect(contour)
                # Person-like aspect ratio
                if 0.3 < (h / w) < 3.0 and w > 30 and h > 60:
                    valid_detections.append(((x, y, w, h), area))
        
        return valid_detections if valid_detections else None
    except Exception as e:
        print(f"[DEBUG] Motion contour detection error: {e}")
    
    return None

def detect_human_pose(frame):
    """Detect human using MediaPipe pose estimation with improved filtering"""
    if not HAS_MEDIAPIPE or pose_detector is None:
        return None
    
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb)
        if not results.pose_landmarks:
            return None
        
        landmarks = results.pose_landmarks.landmark
        
        # Filter landmarks by visibility - require higher confidence
        visible_landmarks = [l for l in landmarks if l.visibility > 0.4]
        if len(visible_landmarks) < 8:  # Need at least 8 visible landmarks
            return None
        
        # Get bounding box from visible landmarks
        x_coords = [l.x for l in visible_landmarks]
        y_coords = [l.y for l in visible_landmarks]
        
        if not x_coords or not y_coords:
            return None
        
        h, w = frame.shape[:2]
        x_min = max(0, int(min(x_coords) * w) - 5)
        x_max = min(w, int(max(x_coords) * w) + 5)
        y_min = max(0, int(min(y_coords) * h) - 5)
        y_max = min(h, int(max(y_coords) * h) + 5)
        
        bbox_w = max(1, x_max - x_min)
        bbox_h = max(1, y_max - y_min)
        
        # Stricter size requirements for better detection
        if bbox_w > 20 and bbox_h > 40 and bbox_h / bbox_w < 3.5:
            # Calculate confidence based on landmarks
            avg_visibility = sum(l.visibility for l in visible_landmarks) / len(visible_landmarks)
            return ((x_min, y_min, bbox_w, bbox_h), bbox_w * bbox_h, avg_visibility)
    except Exception as e:
        print(f"[DEBUG] Pose detection error: {e}")
    
    return None


def detect_person_hog(frame):
    """Detect person using HOG descriptor (human-specific, fast)"""
    if not HAS_CV2:
        return None
    
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Multi-scale detection for robustness
        rects, weights = hog.detectMultiScale(
            gray, 
            winStride=(6, 6), 
            padding=(8, 8), 
            scale=1.05,
            useMeanshiftGrouping=True  # Better grouping
        )
        
        valid_detections = []
        for (x, y, w, h) in rects:
            # Stricter person-like proportions
            if w > 35 and h > 90 and 0.3 < (h / w) < 2.5:
                valid_detections.append(((x, y, w, h), w * h))
        
        return valid_detections if valid_detections else None
    except Exception as e:
        print(f"[DEBUG] HOG detection error: {e}")
    
    return None


def estimate_pose_angle(frame):
    """Estimate torso angle; 0 = upright, 90 = horizontal."""
    if not HAS_MEDIAPIPE or pose_detector is None:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(rgb)
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark
    required = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]
    for idx in required:
        if landmarks[idx].visibility < 0.3:
            return None

    lx = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2
    ly = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
    hx = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2
    hy = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2

    dx = lx - hx
    dy = hy - ly  # # positive when torso points downwards
    if dx == 0 and dy == 0:
        return None
    angle_deg = np.degrees(np.arccos(np.clip((dy / np.sqrt(dx*dx + dy*dy)), -1.0, 1.0)))

    # 0 degrees = perfectly vertical upright, 90 = horizontal
    return float(angle_deg)


@app.route('/')
def login():
    return render_template('login.html')

@app.route('/loginproc', methods=['POST'])
def loginproc():
    username = request.form.get('username')
    password = request.form.get('password')
    # Simple demo login - accept any username/password
    if username and password:
        return redirect(url_for('home'))
    else:
        return redirect(url_for('login'))

@app.route('/home')
def home():
    return render_template('home.html', has_cv2=HAS_CV2)

@app.route('/fall_records_page')
def fall_records_page():
    return render_template('fall_records.html')

# --- KEEP ALL YOUR IMPORTS AND CONFIG SAME ABOVE ---

def analyze_frame(frame):
    """Multi-method person detection with improved fall validation"""
    global fall_alert, alert_time, fall_count, no_fall_count, previous_frame, email_sent_for_current_alert, fall_event_active, potential_fall_alert, potential_fall_time, potential_email_attempted

    if frame is None or frame.size == 0:
        return False, frame, 0.0
    
    h, w = frame.shape[:2]
    is_falling = False
    confidence = 0
    display_text = "Initializing..."
    box_color = (100, 165, 255)

    person_found = False
    x = y = bw = bh = detection_confidence = 0.0

    # DETECTION CASCADE - Try multiple methods for robust detection
    
    # Method 1: MediaPipe Pose (most accurate, best for pose estimation)
    pose_detection = detect_human_pose(frame)
    if pose_detection:
        if len(pose_detection) == 3:
            (x, y, bw, bh), area, det_conf = pose_detection
            detection_confidence = det_conf
        else:
            (x, y, bw, bh), area = pose_detection
            detection_confidence = 0.8
        person_found = True
        print(f"[DETECTION] MediaPipe detected person at ({x}, {y}), size {bw}x{bh}, conf: {detection_confidence:.2f}")

    # Method 2: HOG Descriptor (fast, reliable)
    if not person_found:
        hog_detections = detect_person_hog(frame)
        if hog_detections:
            hog_detections.sort(key=lambda d: d[1], reverse=True)  # Sort by area
            (x, y, bw, bh), area = hog_detections[0]
            detection_confidence = 0.7
            person_found = True
            print(f"[DETECTION] HOG detected person at ({x}, {y}), size {bw}x{bh}")

    # Method 3: Motion Contours (motion-based detection)
    if not person_found and previous_frame is not None:
        motion_detections = detect_motion_contours(frame, previous_frame)
        if motion_detections:
            motion_detections.sort(key=lambda d: d[1], reverse=True)  # Sort by area
            (x, y, bw, bh), area = motion_detections[0]
            detection_confidence = 0.6
            person_found = True
            print(f"[DETECTION] Motion detected person at ({x}, {y}), size {bw}x{bh}")

    # No person case
    if not person_found or bw <= 0 or bh <= 0:
        fall_count = 0
        no_fall_count += 1

        if fall_event_active and no_fall_count >= 18:
            fall_event_active = False
            email_sent_for_current_alert = False
            print("[ALERT] Fall event ended - person no longer detected")

        display_text = "Waiting for person..."
        box_color = (0, 165, 255)

        previous_frame = frame.copy()
        return False, frame, 0.0

    # ---------------- DATA TRACKING ---------------- #
    height_history.append(bh)
    width_history.append(bw)

    pos_y = float(y + bh / 2) / h
    position_history.append(pos_y)

    aspect_ratio = float(bh) / float(bw) if bw > 0 else 1.0

    vertical_speed = 0.0
    if len(position_history) >= 5:
        vertical_speed = max(0.0, position_history[-1] - position_history[-5])

    pose_angle = estimate_pose_angle(frame)

    

    score = 0
    max_score = 10  # Increased from 6 for better granularity

   
    is_horizontal = aspect_ratio < 0.85
    if is_horizontal:
        score += 3

    # Low body position
    is_low_body = pos_y > 0.65
    if is_low_body:
        score += 2

    # Height drop - rapid size change
    height_drop = False
    if len(height_history) >= 6:
        recent_heights = list(height_history)[-6:]
        max_h = np.max(recent_heights)
        min_h = np.min(recent_heights)

        if max_h > 0 and ((max_h - min_h) / max_h) > 0.12:
            score += 2
            height_drop = True

    # Fast downward motion
    fast_drop = vertical_speed > 0.08
    if fast_drop:
        score += 1.5

    # Pose angle
    pose_fall = False
    if pose_angle is not None and pose_angle > 40:
        score += 1.5
        pose_fall = True

    # Stability (lying still)
    stable_fall = False
    if len(position_history) >= 6:
        recent_pos = list(position_history)[-6:]
        if np.std(recent_pos) < 0.04 and is_low_body:
            score += 1
            stable_fall = True

    # Recent large motion
    recent_motion = False
    if len(position_history) >= 8:
        motion_diff = abs(position_history[-1] - position_history[-8])
        if motion_diff > 0.12:
            score += 0.5
            recent_motion = True

    confidence = min(1.0, score / max_score)
    confidence_history.append(confidence)
    print(f"[SCORE] H:{is_horizontal} L:{is_low_body} D:{height_drop} F:{fast_drop} P:{pose_fall} S:{stable_fall} | Score: {score:.1f}/{max_score} Conf: {confidence:.2f}")

    # ---------------- STRICT VALIDATION ---------------- #

    is_currently_fallen = is_horizontal and is_low_body

    fall_transition = (
        height_drop or fast_drop or pose_fall
    )

    fall_transition_history.append(
        1 if is_currently_fallen and fall_transition else 0
    )

    upright_history.append(
        1 if aspect_ratio > 0.8 and (pose_angle is None or pose_angle < 45) else 0
    )

    # Adjust thresholds based on detection method confidence
    threshold_adjustment = (1.0 - detection_confidence) * 0.05  # Lower threshold if higher confidence detection
    adjusted_fall_threshold = max(0.50, FALL_CONFIDENCE_THRESHOLD - threshold_adjustment)
    
    recent_upright = sum(list(upright_history)[-10:]) >= 1  # Even more lenient
    sustained_fall = sum(list(fall_transition_history)[-6:]) >= 2  # Faster detection (was 3)

    # Improved fall validation with better state machine
    is_confident_fall = (
        confidence >= adjusted_fall_threshold and
        sustained_fall and
        (is_horizontal and is_low_body) and
        (height_drop or fast_drop or pose_fall or stable_fall)  # At least one transition factor
    )

    # Alert for potential fall
    is_potential_fall = confidence >= POTENTIAL_FALL_THRESHOLD and not is_confident_fall
    
    if is_potential_fall:
        potential_fall_alert = True
        potential_fall_time = time.time()

    # ---------------- EVENT TRIGGER ---------------- #

    if is_confident_fall:
        fall_count += 1
        no_fall_count = 0
    else:
        no_fall_count += 1
        fall_count = max(0, fall_count - 1)

    # Record and alert on confirmed fall
    if is_confident_fall and not fall_event_active:
        fall_event_active = True
        add_fall_record(confidence, x, y, bw, bh)
        
        # Log detailed fall detection info
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{'='*60}")
        print(f"[FALL DETECTED] {timestamp}")
        print(f"  Confidence: {confidence:.2f} ({int(confidence*100)}%)")
        print(f"  Position: X={x}, Y={y}")
        print(f"  Size: {bw}x{bh}px | Aspect Ratio: {aspect_ratio:.2f}")
        print(f"  Pose Angle: {pose_angle:.1f}° if pose_angle else 'N/A'")
        print(f"  Factors: Horizontal={is_horizontal}, Low={is_low_body}, Drop={height_drop}, Fast={fast_drop}")
        print(f"{'='*60}\n")
        
        # Send one email for each confirmed fall event.
        if email_alert_enabled and not email_sent_for_current_alert:
            body = f"""
CRITICAL FALL ALERT
{'-'*50}

Status: CONFIRMED FALL DETECTED
Time: {timestamp}
Confidence: {int(confidence * 100)}%

Location: X={x}, Y={y}
Size: {bw}x{bh}px
Aspect Ratio: {aspect_ratio:.2f}
Pose Angle: {f'{pose_angle:.1f} degrees' if pose_angle is not None else 'N/A'}

Detection Factors:
- Horizontal Orientation: {'YES' if is_horizontal else 'NO'}
- Low Position: {'YES' if is_low_body else 'NO'}
- Height Drop: {'YES' if height_drop else 'NO'}
- Fast Motion: {'YES' if fast_drop else 'NO'}
- Stable Position: {'YES' if stable_fall else 'NO'}

IMMEDIATE ACTION REQUIRED!
Please check on the person right away and verify their safety.

This is an automated alert from the Fall Detection System.
Please do not reply to this email.
            """
            if send_fall_email(
                'CRITICAL: FALL DETECTED - IMMEDIATE ACTION REQUIRED',
                body
            ):
                email_sent_for_current_alert = True

    # Send at most one warning email during a continuous potential-fall episode.
    if is_potential_fall and not fall_event_active:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[POTENTIAL FALL WARNING] {timestamp} - Confidence: {int(confidence * 100)}%\n")
        if email_alert_enabled and not potential_email_attempted:
            body = f"""
POTENTIAL FALL WARNING
{'-'*50}

Status: POTENTIAL FALL DETECTED
Time: {timestamp}
Confidence: {int(confidence * 100)}%

Please verify the person's status and check on them if needed.

This is an automated alert from the Fall Detection System.
            """
            send_fall_email('WARNING: Potential Fall - Please Verify', body)
            potential_email_attempted = True

    if fall_event_active:
        is_falling = True
        fall_alert = True
        alert_time = time.time()

    # Display status
    if fall_event_active:
        display_text = f"🚨 FALL DETECTED ({int(confidence * 100)}%)"
        box_color = (0, 0, 255)  # Red for confirmed fall
    elif is_potential_fall:
        display_text = f"⚠️  FALL DETECTED ({int(confidence * 100)}%)"
        box_color = (0, 165, 255)  # Orange for potential
    elif confidence >= POTENTIAL_FALL_THRESHOLD:
        display_text = f"⚠️ ALERT ({int(confidence * 100)}%)"
        box_color = (0, 165, 255)  # Orange for monitoring
    else:
        display_text = "✓ NORMAL"
        box_color = (0, 255, 0)  # Green for normal

    if fall_event_active and no_fall_count >= 15:
        fall_event_active = False
        email_sent_for_current_alert = False

    # Draw box
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), box_color, 3)

    pose_text = f"{pose_angle:.1f}°" if pose_angle is not None else "N/A"

    cv2.putText(frame, display_text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

    cv2.putText(frame,
                f"AR:{aspect_ratio:.2f} Pose:{pose_text} Conf:{confidence:.2f}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1)

    previous_frame = frame.copy()
    return is_falling, frame, confidence

def gen_frames():
    """Generate video frames"""
    if not HAS_CV2:
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check if camera is available.")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to read frame from camera")
                break
            
            frame = cv2.resize(frame, (640, 480))
            
            try:
                is_fall, frame, conf = analyze_frame(frame)
            except Exception as e:
                print(f"[ERROR] Frame analysis failed: {e}")
            
            try:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"[ERROR] Frame encoding failed: {e}")
    except Exception as e:
        print(f"[ERROR] Video stream error: {e}")
    finally:
        cap.release()

@app.route('/video_feed')
def video_feed():
    # If OpenCV isn't available, return a friendly message instead of aborting
    if not HAS_CV2:
        return Response('Video unavailable: OpenCV (cv2) is not installed on this system.', mimetype='text/plain')

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/fall_status')
def fall_status():
    """Check fall status - returns both confirmed and potential falls"""
    global fall_alert, alert_time, potential_fall_alert, potential_fall_time, email_sent_for_current_alert, potential_email_attempted
    
    # Reset confirmed fall alert after 5 seconds
    if fall_alert and time.time() - alert_time > 5:
        fall_alert = False
        email_sent_for_current_alert = False
    
    # Reset potential fall alert after 3 seconds
    if potential_fall_alert and time.time() - potential_fall_time > 3:
        potential_fall_alert = False
        potential_email_attempted = False
    
    return jsonify({
        'fall_detected': fall_alert,
        'potential_fall': potential_fall_alert
    })


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Get / update runtime fall detection settings"""
    global email_alert_enabled

    if request.method == 'GET':
        return jsonify({
            'fall_detection_mode': 'AUTOMATIC',
            'fall_confidence_threshold': FALL_CONFIDENCE_THRESHOLD,
            'potential_fall_threshold': POTENTIAL_FALL_THRESHOLD,
            'email_alert_enabled': email_alert_enabled
        })

    # POST updates
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'error': 'No settings payload provided'}), 400

    if 'email_alert_enabled' in data:
        email_alert_enabled = bool(data['email_alert_enabled'])

    return jsonify({
        'fall_detection_mode': 'AUTOMATIC',
        'fall_confidence_threshold': FALL_CONFIDENCE_THRESHOLD,
        'potential_fall_threshold': POTENTIAL_FALL_THRESHOLD,
        'email_alert_enabled': email_alert_enabled,
        'message': 'Settings updated'
    })


@app.route('/fall_records/since/<int:minutes>')
def fall_records_since(minutes):
    """Return fall records in the last N minutes"""
    load_fall_records()
    cutoff = datetime.now() - timedelta(minutes=minutes)
    filtered = [r for r in fall_records if datetime.fromisoformat(r['timestamp']) >= cutoff]
    return jsonify({'minutes': minutes, 'count': len(filtered), 'records': filtered})


def compute_fall_summary(records):
    """Return additional fall summary statistics."""
    total = len(records)
    now = datetime.now()
    one_day_ago = now - timedelta(days=1)
    recent_count = len([r for r in records if datetime.fromisoformat(r['timestamp']) > one_day_ago])
    high_severity = len([r for r in records if r.get('severity') == 'HIGH'])
    avg_confidence = float(sum(r.get('confidence', 0) for r in records) / total) if total else 0.0
    last_fall_time = records[-1]['timestamp'] if total else None

    if last_fall_time:
        last_fall_dt = datetime.fromisoformat(last_fall_time)
        minutes_since_last = max(0, (now - last_fall_dt).total_seconds() / 60)
        since_last = f"{minutes_since_last:.1f}"
    else:
        minutes_since_last = None
        since_last = 'N/A'

    risk_ratio = (high_severity / total) if total else 0
    if risk_ratio >= 0.50:
        risk_level = 'HIGH'
    elif risk_ratio >= 0.20:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'LOW'

    return {
        'total_falls': total,
        'recent_24h': recent_count,
        'high_severity': high_severity,
        'average_confidence': avg_confidence,
        'last_fall_time': last_fall_time,
        'minutes_since_last_fall': float(f"{minutes_since_last:.1f}") if last_fall_time else None,
        'risk_level': risk_level,
    }


def compute_fall_trend(records, days=7):
    """Return daily fall counts for the last N days."""
    now = datetime.now().date()
    trend = []

    for offset in range(days - 1, -1, -1):
        day = now - timedelta(days=offset)
        day_count = sum(1 for r in records if datetime.fromisoformat(r['timestamp']).date() == day)
        trend.append({'date': day.isoformat(), 'count': day_count})

    average_daily = float(sum(item['count'] for item in trend) / days) if days else 0.0
    return {'trend': trend, 'average_daily': average_daily}


@app.route('/fall_records/summary')
def fall_records_summary():
    """Return fall records summary"""
    load_fall_records()
    summary = compute_fall_summary(fall_records)
    return jsonify(summary)


@app.route('/fall_records/trend')
def fall_records_trend():
    """Return fall records trend"""
    load_fall_records()
    trend = compute_fall_trend(fall_records)
    return jsonify(trend)


@app.route('/fall_records', methods=['GET', 'POST'])
def get_fall_records():
    if request.method == 'POST':
        # Add a new record
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        confidence = data.get('confidence')
        x = data.get('x')
        y = data.get('y')
        width = data.get('width')
        height = data.get('height')
        
        if not all([confidence is not None, x is not None, y is not None, width is not None, height is not None]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        try:
            confidence = float(confidence)
            x = int(x)
            y = int(y)
            width = int(width)
            height = int(height)
        except ValueError:
            return jsonify({'error': 'Invalid data types'}), 400
        
        record = add_fall_record(confidence, x, y, width, height)
        return jsonify(record)
    
    # GET all records
    load_fall_records()
    return jsonify(fall_records)


@app.route('/fall_records/<int:record_id>', methods=['DELETE'])
def delete_fall_record(record_id):
    """Delete a specific fall record"""
    global fall_records
    load_fall_records()
    fall_records = [r for r in fall_records if r.get('id') != record_id]
    save_fall_records()
    return jsonify({'message': f'Record {record_id} deleted'})


@app.route('/fall_records/clear', methods=['POST'])
def clear_fall_records():
    """Clear all fall records"""
    global fall_records
    fall_records = []
    save_fall_records()
    return jsonify({'message': 'All records cleared'})


@app.route('/fall_records/export/csv', methods=['GET'])
def export_fall_records_csv():
    """Export fall records as CSV"""
    load_fall_records()
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['ID', 'Timestamp', 'Confidence', 'Severity', 'Location X', 'Location Y', 'Width', 'Height'])
    for record in fall_records:
        loc = record.get('location', {})
        cw.writerow([
            record.get('id'),
            record.get('timestamp'),
            record.get('confidence'),
            record.get('severity'),
            loc.get('x'),
            loc.get('y'),
            loc.get('width'),
            loc.get('height')
        ])
    output = si.getvalue()
    si.close()
    return Response(output, mimetype='text/csv', headers={'Content-Disposition': 'attachment; filename=fall_records.csv'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
