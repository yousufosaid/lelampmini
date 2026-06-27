import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Standard 3D model points of a generic face (in world coordinates)
# Used to map 2D landmarks back to 3D space to find rotation
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

# MediaPipe landmark indices corresponding to the MODEL_POINTS above
LANDMARK_INDICES = [1, 199, 33, 263, 61, 291]

def get_head_pose(landmarks, img_w, img_h):
    # Extract the 2D pixel coordinates for the required landmarks
    image_points = []
    for idx in LANDMARK_INDICES:
        lm = landmarks.landmark[idx]
        image_points.append([lm.x * img_w, lm.y * img_h])
    image_points = np.array(image_points, dtype=np.float64)

    # Camera intrinsic parameters (approximated based on frame size)
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion

    # Solve PnP to get rotation vector (rvec) and translation vector (tvec)
    success, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    if not success:
        return None, None

    # Convert rotation vector to rotation matrix, then to Euler angles
    rmat, _ = cv2.Rodrigues(rvec)
    proj_matrix = np.hstack((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    
    pitch = euler_angles[0, 0]
    yaw = euler_angles[1, 0]
    
    return pitch, yaw

def main():
    cap = cv2.VideoCapture(0)
    
    # Engagement thresholds (in degrees)
    YAW_THRESHOLD = 30      # Comfortably allows looking slightly left/right
    PITCH_MIN = -180        # Lower bound for your camera's pitch inversion
    PITCH_MAX = -150
    
    engagement_state = "DISENGAGED"
    last_state_change = time.time()

    print("--- Starting Perception Stream ---")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip horizontally for natural mirror view, convert to RGB
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(rgb_frame)
        
        current_gaze_engaged = False
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                pitch, yaw = get_head_pose(face_landmarks, w, h)
                
                if pitch is not None and yaw is not None:
                    # Check if head orientation matches your calibrated safe zone
                    if abs(yaw) < YAW_THRESHOLD and (PITCH_MIN <= pitch <= PITCH_MAX):
                        current_gaze_engaged = True
                    
                    # Draw visual feedback on frame
                    cv2.putText(frame, f"Pitch: {int(pitch)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Yaw: {int(yaw)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Basic hysteresis state logic to prevent flickering states
        if current_gaze_engaged and engagement_state == "DISENGAGED":
            engagement_state = "ENGAGED"
            print(f"[{time.strftime('%X')}] State Event -> ENGAGED")
        elif not current_gaze_engaged and engagement_state == "ENGAGED":
            # 1.5-second timeout grace period before officially disengaging
            if time.time() - last_state_change > 1.5:
                engagement_state = "DISENGAGED"
                print(f"[{time.strftime('%X')}] State Event -> DISENGAGED")
        else:
            last_state_change = time.time()

        # Display state on screen
        color = (0, 255, 0) if engagement_state == "ENGAGED" else (0, 0, 255)
        cv2.putText(frame, f"STATUS: {engagement_state}", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        cv2.imshow("LeLamp Perception Node", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()