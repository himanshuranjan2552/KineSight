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

def get_joint_angle(landmarks, a_id, b_id, c_id):
    """
    Calculates the interior angle at joint b_id formed by a_id-b_id-c_id.
    Returns angle in degrees.
    """
    pa = np.array([landmarks[a_id]['x'], landmarks[a_id]['y']])
    pb = np.array([landmarks[b_id]['x'], landmarks[b_id]['y']])
    pc = np.array([landmarks[c_id]['x'], landmarks[c_id]['y']])

    v1 = pa - pb
    v2 = pc - pb
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0

    cosine = np.dot(v1, v2) / (norm_v1 * norm_v2)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def get_upper_arm_vertical_angle(landmarks):
    """
    Angle of the right upper arm (shoulder to elbow) from vertical.
    Used to detect upper arm swing during bicep curls.
    """
    # Right shoulder: 12, Right elbow: 14
    p1 = np.array([landmarks[12]['x'], landmarks[12]['y']])  # shoulder
    p2 = np.array([landmarks[14]['x'], landmarks[14]['y']])  # elbow
    vertical_pt = np.array([p2[0], 0])

    v1 = p1 - p2
    v2 = vertical_pt - p2
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0

    cosine = np.dot(v1, v2) / (norm_v1 * norm_v2)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


# ---------------------------------------------------------------------------
# Push-up
# ---------------------------------------------------------------------------

def determine_pushup_state(elbow_angle):
    """
    s1: Up position (arms extended), s2: Transition, s3: Bottom (arms bent).
    Angle measured at elbow: right shoulder - right elbow - right wrist.
    """
    if elbow_angle >= config.PUSHUP_S1_MIN:
        return "s1"
    elif elbow_angle <= config.PUSHUP_S3_MAX:
        return "s3"
    return "s2"


def get_pushup_feedback(landmarks):
    """
    Checks body alignment during a push-up by comparing hip y-position
    to the midpoint of shoulder and ankle.
    """
    shoulder_y = landmarks[12]['y']
    hip_y = landmarks[24]['y']
    ankle_y = landmarks[28]['y']
    midpoint_y = (shoulder_y + ankle_y) / 2

    if hip_y > midpoint_y + config.PUSHUP_BODY_SAG_THRESH:
        return "Hips sagging! Engage core."
    if hip_y < midpoint_y - config.PUSHUP_BODY_RAISE_THRESH:
        return "Lower your hips!"
    return "Good Form"


# ---------------------------------------------------------------------------
# Bicep Curl
# ---------------------------------------------------------------------------

def determine_curl_state(elbow_angle):
    """
    s1: Rest/arm extended, s2: Transition, s3: Top/arm fully curled.
    Angle measured at elbow: right shoulder - right elbow - right wrist.
    """
    if elbow_angle >= config.CURL_S1_MIN:
        return "s1"
    elif elbow_angle <= config.CURL_S3_MAX:
        return "s3"
    return "s2"


def get_curl_feedback(upper_arm_angle):
    """
    Detects upper arm swing during a bicep curl.
    upper_arm_angle: angle of right shoulder-to-elbow segment from vertical.
    """
    if upper_arm_angle > config.CURL_UPPER_ARM_SWING:
        return "Keep elbow at your side!"
    return "Good Form"


# ---------------------------------------------------------------------------
# Plank
# ---------------------------------------------------------------------------

def get_plank_feedback(landmarks):
    """
    Checks plank body alignment using hip position relative to the
    shoulder-to-ankle midpoint (normalized y-coordinates).
    """
    shoulder_y = landmarks[12]['y']
    hip_y = landmarks[24]['y']
    ankle_y = landmarks[28]['y']
    midpoint_y = (shoulder_y + ankle_y) / 2

    if hip_y > midpoint_y + config.PLANK_BODY_SAG_THRESH:
        return "Hips sagging! Engage core."
    if hip_y < midpoint_y - config.PLANK_BODY_RAISE_THRESH:
        return "Lower your hips!"
    return "Good plank form! Hold it."


# ---------------------------------------------------------------------------
# Squat (original)
# ---------------------------------------------------------------------------

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