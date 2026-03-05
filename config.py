# --- View Validation ---
OFFSET_THRESH = 15.0  # Max allowed nose offset before flagging frontal view

# --- State Thresholds (Knee-Vertical Angle) ---
STATE_S1_MAX = 32.0   # s1: Normal/Standing phase
STATE_S2_MIN = 35.0   # s2: Transition phase start
STATE_S2_MAX = 65.0   # s2: Transition phase end
STATE_S3_MIN = 75.0   # s3: Pass phase start
STATE_S3_MAX = 95.0   # s3: Pass phase end

# --- Feedback Thresholds ---
FEEDBACK_BEND_FORWARD = 20.0  # Shoulder-Hip angle < 20
FEEDBACK_BEND_BACKWARD = 45.0 # Shoulder-Hip angle > 45
FEEDBACK_KNEE_OVER_TOES = 30.0 # Knee-Ankle angle > 30
FEEDBACK_DEEP_SQUAT = 95.0     # Hip-Knee angle > 95

# --- Timing & Inactivity ---
INACTIVE_THRESH = 15.0 # Seconds before resetting counters

# --- Exercise Mode ---
# Beginner mode typically uses more relaxed thresholds
# Pro mode requires stricter adherence to form
MODE = "Beginner"