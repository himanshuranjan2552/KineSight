import cv2
import time
import pose_engine
import utils
import config

def run_trainer():
    # 1. Initialization
    cap = cv2.VideoCapture(0)  # Use 0 for webcam [cite: 52]
    engine = pose_engine.PoseEngine()
    
    # Global state variables [cite: 51, 52]
    correct_count = 0
    incorrect_count = 0
    state_sequence = []
    current_state = "s1"
    last_detection_time = time.time()
    
    print("AI Fitness Trainer Started. Please stand in a SIDE VIEW.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 2. Extract Landmarks [cite: 53]
        frame, landmarks = engine.get_landmarks(frame)

        if landmarks:
            last_detection_time = time.time()  # Reset inactivity timer [cite: 54, 87]
            
            # 3. Side-View Validation (Offset Angle) [cite: 57, 148]
            offset_angle = utils.calculate_offset(landmarks)
            if offset_angle > config.OFFSET_THRESH:
                cv2.putText(frame, "WARNING: FACING CAMERA. PLEASE TURN SIDEWAYS", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # 4. Geometry Calculations [cite: 59, 101]
                hip_angle = utils.get_vertical_angle(landmarks, "hip")
                knee_angle = utils.get_vertical_angle(landmarks, "knee")
                ankle_angle = utils.get_vertical_angle(landmarks, "ankle")  # Add this line
                
                # 5. State Machine Logic [cite: 33, 60]
                new_state = utils.determine_state(knee_angle)
                
                if new_state != current_state:
                    if new_state == "s1":
                        # Transition back to s1: Evaluate the completed squat [cite: 62]
                        completed_rep = (
                            len(state_sequence) >= 3
                            and state_sequence[0] == "s2"
                            and "s3" in state_sequence
                            and state_sequence[-1] == "s2"
                        )

                        if completed_rep:
                            correct_count += 1
                        elif len(state_sequence) > 0:
                            incorrect_count += 1
                        state_sequence = []  # Reset sequence [cite: 44, 45]
                    else:
                        # Record state changes in order; allow returning to s2 after s3.
                        # Prevent duplicate consecutive entries caused by noisy frames.
                        if not state_sequence or state_sequence[-1] != new_state:
                            state_sequence.append(new_state)
                    
                    current_state = new_state

                # 6. Generate Feedback [cite: 63, 149]
                feedback = utils.get_feedback(hip_angle, knee_angle, ankle_angle, current_state)
                cv2.putText(frame, f"Feedback: {feedback}", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 7. Check for Inactivity [cite: 55, 82, 84]
        if time.time() - last_detection_time > config.INACTIVE_THRESH:
            correct_count = 0
            incorrect_count = 0
            cv2.putText(frame, "RESETTING DUE TO INACTIVITY", (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 8. UI Overlay
        cv2.putText(frame, f"Correct: {correct_count}", (50, 400), 1, 2, (0, 255, 0), 2)
        cv2.putText(frame, f"Incorrect: {incorrect_count}", (50, 450), 1, 2, (0, 0, 255), 2)
        
        cv2.imshow('KineSight AI Trainer', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_trainer()