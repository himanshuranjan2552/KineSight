import numpy as np
import config

def get_vertical_angle(landmarks, joint_type):
    """
    Calculates the angle of a specific body line with the vertical axis.
    Logic: Angle between (Point A, Point B) and (Point B, Vertical Point).
    """
    # MediaPipe Landmark IDs:
    # Right Shoulder: 12 | Right Hip: 24 | Right Knee: 26 | Right Ankle: 28
    
    if joint_type == "hip":
        # Calculate angle of Shoulder-Hip line with the vertical
        p1 = np.array([landmarks[12]['x'], landmarks[12]['y']])
        p2 = np.array([landmarks[24]['x'], landmarks[24]['y']])
    elif joint_type == "knee":
        # Calculate angle of Hip-Knee line with the vertical
        p1 = np.array([landmarks[24]['x'], landmarks[24]['y']])
        p2 = np.array([landmarks[26]['x'], landmarks[26]['y']])
    elif joint_type == "ankle":
        # Calculate angle of Knee-Ankle line with the vertical
        p1 = np.array([landmarks[26]['x'], landmarks[26]['y']])
        p2 = np.array([landmarks[28]['x'], landmarks[28]['y']])
    else:
        return 0

    # Create a vertical reference point directly above the vertex (p2)
    # Since y-axis increases downwards in image space, y=0 is "up"
    vertical_pt = np.array([p2[0], 0]) 
    
    # Vector Calculation: v1 (the limb), v2 (the vertical)
    v1 = p1 - p2
    v2 = vertical_pt - p2
    
    # Calculate cosine of angle using dot product
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
        
    cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    
    return angle

def calculate_offset(landmarks):
    """
    Calculates the offset distance between nose and shoulders.
    Used to verify the user is in SIDE VIEW.
    """
    # Nose: 0 | Left Shoulder: 11 | Right Shoulder: 12
    nose_x = landmarks[0]['x']
    l_sh_x = landmarks[11]['x']
    r_sh_x = landmarks[12]['x']
    
    # Distance of nose from the shoulder midpoint
    mid_shoulder_x = (l_sh_x + r_sh_x) / 2
    dist = abs(nose_x - mid_shoulder_x)
    
    # Return as percentage-like scale for thresholding
    return dist * 100 

def determine_state(knee_angle):
    """
    Maps the hip-knee vertical angle to the squat state machine.
    s1: Normal (Standing)
    s2: Transition (Descending/Ascending)
    s3: Pass (Bottom of squat)
    """
    if knee_angle < config.STATE_S1_MAX:
        return "s1"
    elif config.STATE_S2_MIN <= knee_angle <= config.STATE_S2_MAX:
        return "s2"
    elif config.STATE_S3_MIN <= knee_angle <= config.STATE_S3_MAX:
        return "s3"
    return "s2" # Default transition state

def get_feedback(hip_angle, knee_angle, ankle_angle, current_state):
    """
    Analyzes angles against heuristic thresholds to provide real-time guidance.
    """
    # 1. Back Posture Feedback
    if hip_angle < config.FEEDBACK_BEND_FORWARD:
        return "Bend Forward"
    if hip_angle > config.FEEDBACK_BEND_BACKWARD:
        return "Bend Backwards"
        
    # 2. Depth Feedback
    if config.STATE_S2_MIN < knee_angle < config.STATE_S2_MAX and current_state == "s2":
        return "Lower your hips"
        
    # 3. Injury Prevention Feedback
    if ankle_angle > config.FEEDBACK_KNEE_OVER_TOES:
        return "Knee over toes! Adjust."
        
    if knee_angle > config.FEEDBACK_DEEP_SQUAT:
        return "Deep Squat Alert!"
        
    return "Good Form"