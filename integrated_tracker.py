import cv2
import dlib
import numpy as np
import os
import torch
from yolov5 import YOLOv5
import math
from collections import deque
import warnings

# Suppress FutureWarning for torch.cuda.amp
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")

# Initialize webcam
def open_webcam():
    for index in range(5):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Webcam opened on index {index}!")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            return cap
        print(f"Camera index {index} failed.")
    print("Error: No webcam found.")
    exit()

# Load detectors and predictors
detector = dlib.get_frontal_face_detector()
predictor_path = os.path.join(os.path.dirname(__file__), "models", "shape_predictor_68_face_landmarks.dat")
model_path = os.path.join(os.path.dirname(__file__), "models", "best.pt")

try:
    predictor = dlib.shape_predictor(predictor_path)
    print("Shape predictor loaded!")
except RuntimeError as e:
    print(f"Error: Could not load shape predictor. Error: {e}")
    exit()

try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mobile_model = YOLOv5(model_path, device=device)
    print("YOLOv5 model loaded!")
except Exception as e:
    print(f"Error: Could not load YOLOv5 model. Error: {e}")
    exit()

# Smoothing parameters
PUPIL_HISTORY_SIZE = 5
ANGLE_HISTORY_SIZE = 5
LANDMARK_HISTORY_SIZE = 5
pupil_history = deque(maxlen=PUPIL_HISTORY_SIZE)
yaw_history = deque(maxlen=ANGLE_HISTORY_SIZE)
pitch_history = deque(maxlen=ANGLE_HISTORY_SIZE)
roll_history = deque(maxlen=ANGLE_HISTORY_SIZE)
landmark_history = deque(maxlen=LANDMARK_HISTORY_SIZE)

# Global variables
previous_head_state = "Looking at Screen"

# Head Pose: Dynamic 3D Model Points
def get_dynamic_model_points(landmarks):
    face_width = abs(landmarks.part(16).x - landmarks.part(0).x)
    face_height = abs(landmarks.part(8).y - landmarks.part(19).y)
    scale = face_width / 60.0
    return np.array([
        (0.0, 0.0, 0.0),                     # Nose tip
        (0.0, -50.0 * scale, -10.0 * scale), # Chin
        (-30.0 * scale, 40.0 * scale, -10.0 * scale), # Left eye
        (30.0 * scale, 40.0 * scale, -10.0 * scale),  # Right eye
        (-25.0 * scale, -30.0 * scale, -10.0 * scale), # Left mouth corner
        (25.0 * scale, -30.0 * scale, -10.0 * scale)   # Right mouth corner
    ], dtype=np.float64)

# Head Pose: Camera Calibration
def get_camera_matrix(frame_width):
    focal_length = frame_width
    center = (frame_width / 2, frame_width * 3 / 8)
    return np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64), np.array([0.1, 0.1, 0.0, 0.0], dtype=np.float64).reshape(4, 1)

# Head Pose: Calculate Angles
def get_head_pose_angles(image_points, model_points, camera_matrix, dist_coeffs):
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None, None, None, None, None
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0
    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll), rotation_vector, translation_vector

# Smooth angles and landmarks
def smooth_angle(angle_history, new_angle):
    angle_history.append(new_angle)
    return np.median(angle_history)

def smooth_landmarks(new_points):
    landmark_history.append(new_points)
    return np.mean(landmark_history, axis=0)

# Pupil Detection
def detect_pupil(eye_region, frame_count):
    if eye_region is None or eye_region.size == 0:
        print(f"Frame {frame_count}: Empty eye region!")
        return None
    if len(eye_region.shape) == 3:
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    else:
        gray_eye = eye_region
    gray_eye = cv2.equalizeHist(gray_eye)
    blurred_eye = cv2.GaussianBlur(gray_eye, (5, 5), 0)
    _, threshold_eye = cv2.threshold(blurred_eye, 40, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(f"debug_eye_{frame_count}.png", gray_eye)
    cv2.imwrite(f"debug_threshold_{frame_count}.png", threshold_eye)
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10 and area < (eye_region.shape[0] * eye_region.shape[1]) * 0.08:
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                contour_img = cv2.cvtColor(threshold_eye, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 1)
                cv2.imwrite(f"debug_contour_{frame_count}.png", contour_img)
                return (int(cx), int(cy))
        print(f"Frame {frame_count}: No valid pupil contour!")
    else:
        print(f"Frame {frame_count}: No contours!")
    return None

# Mobile Detection
def process_mobile_detection(frame, frame_count):
    results = mobile_model.predict(frame, size=640, augment=False)
    mobile_detected = False
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        conf = conf.item()
        cls = int(cls.item())
        if conf < 0.5 or cls != 0:
            print(f"Frame {frame_count}: Missed/Rejected detection: Confidence {conf:.2f}, Class {cls}")
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        w = x2 - x1
        h = y2 - y1
        aspect_ratio = w / h if h != 0 else 0
        if aspect_ratio < 0.3 or aspect_ratio > 3:
            print(f"Frame {frame_count}: False positive rejected: Aspect ratio {aspect_ratio:.2f}")
            continue
        label = f"Mobile ({conf:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        mobile_detected = True
    return frame, mobile_detected

# Draw Pose Axes
def draw_pose_axes(frame, rotation_vector, translation_vector, camera_matrix, dist_coeffs):
    points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
    projected_points, _ = cv2.projectPoints(points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    projected_points = projected_points.astype(np.int32)
    origin = tuple(projected_points[3].ravel())
    cv2.line(frame, origin, tuple(projected_points[0].ravel()), (0, 0, 255), 3)  # X-axis (red)
    cv2.line(frame, origin, tuple(projected_points[1].ravel()), (0, 255, 0), 3)  # Y-axis (green)
    cv2.line(frame, origin, tuple(projected_points[2].ravel()), (255, 0, 0), 3)  # Z-axis (blue)

# Main processing function
def process_frame(frame, frame_count, thresholds=(5, 8, 3)):
    global previous_head_state
    if frame is None or frame.size == 0:
        print("Error: Frame is None or empty!")
        return None, "No Frame", "No Gaze", "No Mobile"
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    camera_matrix, dist_coeffs = get_camera_matrix(frame.shape[1])
    head_direction = "No Head Direction"
    gaze_direction = "No Gaze"
    mobile_status = "No Mobile"
    # Mobile detection
    frame, mobile_detected = process_mobile_detection(frame, frame_count)
    if mobile_detected:
        mobile_status = "Mobile Detected"
    # Face detection
    faces = detector(gray, 1)
    if not faces:
        print(f"Frame {frame_count}: No faces detected!")
        return frame, head_direction, gaze_direction, mobile_status
    face = faces[0]
    landmarks = predictor(gray, face)
    # Head pose processing
    model_points = get_dynamic_model_points(landmarks)
    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),
        (landmarks.part(8).x, landmarks.part(8).y),
        (landmarks.part(36).x, landmarks.part(36).y),
        (landmarks.part(45).x, landmarks.part(45).y),
        (landmarks.part(48).x, landmarks.part(48).y),
        (landmarks.part(54).x, landmarks.part(54).y)
    ], dtype=np.float64)
    image_points = smooth_landmarks(image_points)
    angles = get_head_pose_angles(image_points, model_points, camera_matrix, dist_coeffs)
    if angles[0] is None:
        print(f"Frame {frame_count}: Head pose calculation failed!")
    else:
        pitch, yaw, roll, rotation_vector, translation_vector = angles
        pitch = smooth_angle(pitch_history, pitch)
        yaw = smooth_angle(yaw_history, yaw)
        roll = smooth_angle(roll_history, roll)
        print(f"Frame {frame_count}: Smoothed angles - Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Roll: {roll:.2f}")
        draw_pose_axes(frame, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        pitch_offset, yaw_offset, roll_offset = (0.0, 0.0, 0.0)  # Default offsets
        pitch_threshold, yaw_threshold, roll_threshold = thresholds
        if (abs(yaw - yaw_offset) <= yaw_threshold and 
            abs(pitch - pitch_offset) <= pitch_threshold and 
            abs(roll - roll_offset) <= roll_threshold):
            head_direction = "Looking at Screen"
        elif yaw < yaw_offset - yaw_threshold - 1:
            head_direction = "Looking Left"
        elif yaw > yaw_offset + yaw_threshold + 1:
            head_direction = "Looking Right"
        elif pitch > pitch_offset + pitch_threshold + 1:
            head_direction = "Looking Up"
        elif pitch < pitch_offset - pitch_threshold - 1:
            head_direction = "Looking Down"
        elif abs(roll - roll_offset) > roll_threshold + 1:
            head_direction = "Tilted"
        else:
            head_direction = previous_head_state
        previous_head_state = head_direction
    # Pupil detection
    left_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
    right_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
    left_eye_rect = cv2.boundingRect(left_eye_points)
    right_eye_rect = cv2.boundingRect(right_eye_points)
    padding = 8
    left_eye = frame[max(0, left_eye_rect[1]-padding):left_eye_rect[1] + left_eye_rect[3]+padding,
                     max(0, left_eye_rect[0]-padding):left_eye_rect[0] + left_eye_rect[2]+padding]
    right_eye = frame[max(0, right_eye_rect[1]-padding):right_eye_rect[1] + right_eye_rect[3]+padding,
                      max(0, right_eye_rect[0]-padding):right_eye_rect[0] + right_eye_rect[2]+padding]
    left_pupil = detect_pupil(left_eye, frame_count)
    right_pupil = detect_pupil(right_eye, frame_count)
    print(f"Frame {frame_count}: Left pupil: {left_pupil}, Right pupil: {right_pupil}")
    cv2.rectangle(frame, (left_eye_rect[0], left_eye_rect[1]), 
                  (left_eye_rect[0] + left_eye_rect[2], left_eye_rect[1] + left_eye_rect[3]), (0, 255, 0), 1)
    cv2.rectangle(frame, (right_eye_rect[0], right_eye_rect[1]), 
                  (right_eye_rect[0] + right_eye_rect[2], right_eye_rect[1] + right_eye_rect[3]), (0, 255, 0), 1)
    if left_pupil and right_pupil:
        lx, ly = left_pupil[0] + left_eye_rect[0] - padding, left_pupil[1] + left_eye_rect[1] - padding
        rx, ry = right_pupil[0] + right_eye_rect[0] - padding, right_pupil[1] + right_eye_rect[1] - padding
        pupil_history.append(((lx, ly), (rx, ry)))
        avg_left = np.mean([p[0] for p in pupil_history], axis=0)
        avg_right = np.mean([p[1] for p in pupil_history], axis=0)
        lx, ly = avg_left
        rx, ry = avg_right
        norm_lx = left_pupil[0] / left_eye.shape[1] if left_eye.shape[1] > 0 else 0.5
        norm_rx = right_pupil[0] / right_eye.shape[1] if right_eye.shape[1] > 0 else 0.5
        norm_ly = left_pupil[1] / left_eye.shape[0] if left_eye.shape[0] > 0 else 0.5
        norm_ry = right_pupil[1] / right_eye.shape[0] if right_eye.shape[0] > 0 else 0.5
        print(f"Frame {frame_count}: Norm: Left ({norm_lx:.2f}, {norm_ly:.2f}), Right ({norm_rx:.2f}, {norm_ry:.2f})")
        if norm_ly > 0.45 and norm_ry > 0.45:
            gaze_direction = "Down"
        elif norm_lx < 0.4 and norm_rx < 0.4:
            gaze_direction = "Left"
        elif norm_lx > 0.6 and norm_rx > 0.6:
            gaze_direction = "Right"
        elif norm_ly < 0.4 and norm_ry < 0.4:
            gaze_direction = "Up"
        else:
            gaze_direction = "Straight"
        cv2.putText(frame, f"L: ({norm_lx:.2f}, {norm_ly:.2f})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"R: ({norm_rx:.2f}, {norm_ry:.2f})", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(frame, (int(lx), int(ly)), 3, (0, 0, 255), -1)
        cv2.circle(frame, (int(rx), int(ry)), 3, (0, 0, 255), -1)
    else:
        eye_width, eye_height = left_eye_rect[2], left_eye_rect[3]
        cv2.circle(frame, (left_eye_rect[0] + eye_width//2, left_eye_rect[1] + eye_height//2), 3, (0, 0, 255), -1)
        cv2.circle(frame, (right_eye_rect[0] + eye_width//2, right_eye_rect[1] + eye_height//2), 3, (0, 0, 255), -1)
        print(f"Frame {frame_count}: Pupil detection failed!")
    return frame, head_direction, gaze_direction, mobile_status

def main():
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"dlib Version: {dlib.__version__}")
    print(f"PyTorch Version: {torch.__version__}")
    cap = open_webcam()
    frame_count = 0
    print("Starting integrated tracking...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            continue
        frame, head_direction, gaze_direction, mobile_status = process_frame(frame, frame_count)
        cv2.putText(frame, f"Head: {head_direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Gaze: {gaze_direction}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Mobile: {mobile_status}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Integrated Tracker", frame)
        print(f"Frame {frame_count}: Head: {head_direction}, Gaze: {gaze_direction}, Mobile: {mobile_status}")
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()