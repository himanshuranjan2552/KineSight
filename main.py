import cv2
import time
import pose_engine
import utils
import config

EXERCISES = {
    1: "Squat",
    2: "Push-up",
    3: "Bicep Curl",
    4: "Plank",
}


def show_menu():
    print("\n" + "=" * 46)
    print("        KineSight AI Fitness Trainer")
    print("=" * 46)
    print("Select an exercise:")
    for key, name in EXERCISES.items():
        print(f"  {key}. {name}")
    while True:
        try:
            choice = int(input("\nEnter choice (1-4): "))
            if choice in EXERCISES:
                return choice
        except ValueError:
            pass
        print("Invalid choice. Please enter a number from 1 to 4.")


def run_trainer(exercise):
    cap = cv2.VideoCapture(0)
    engine = pose_engine.PoseEngine()

    correct_count = 0
    incorrect_count = 0
    state_sequence = []
    current_state = "s1"
    last_detection_time = time.time()

    # Plank-only state
    plank_start = None
    hold_time = 0
    best_hold = 0

    exercise_name = EXERCISES[exercise]
    print(f"\n{exercise_name} Trainer Started. Please stand in a SIDE VIEW.")
    print("Press 'q' to quit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Extract landmarks
        frame, landmarks = engine.get_landmarks(frame)

        if landmarks:
            last_detection_time = time.time()

            # 2. Side-view validation
            offset = utils.calculate_offset(landmarks)
            if offset > config.OFFSET_THRESH:
                cv2.putText(frame, "WARNING: FACING CAMERA. PLEASE TURN SIDEWAYS",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            else:
                # 3. Exercise-specific calculations
                if exercise == 1:  # Squat
                    hip_angle = utils.get_vertical_angle(landmarks, "hip")
                    knee_angle = utils.get_vertical_angle(landmarks, "knee")
                    ankle_angle = utils.get_vertical_angle(landmarks, "ankle")
                    new_state = utils.determine_state(knee_angle)
                    feedback = utils.get_feedback(hip_angle, knee_angle, ankle_angle, new_state)

                elif exercise == 2:  # Push-up
                    elbow_angle = utils.get_joint_angle(landmarks, 12, 14, 16)
                    new_state = utils.determine_pushup_state(elbow_angle)
                    feedback = utils.get_pushup_feedback(landmarks)

                elif exercise == 3:  # Bicep Curl
                    elbow_angle = utils.get_joint_angle(landmarks, 12, 14, 16)
                    upper_arm_angle = utils.get_upper_arm_vertical_angle(landmarks)
                    new_state = utils.determine_curl_state(elbow_angle)
                    feedback = utils.get_curl_feedback(upper_arm_angle)

                # 4. State machine for rep-counted exercises (squat / push-up / curl)
                if exercise in (1, 2, 3):
                    if new_state != current_state:
                        if new_state == "s1":
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
                            state_sequence = []
                        else:
                            if not state_sequence or state_sequence[-1] != new_state:
                                state_sequence.append(new_state)
                        current_state = new_state

                    cv2.putText(frame, f"Feedback: {feedback}", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # 5. Plank hold timer
                else:
                    feedback = utils.get_plank_feedback(landmarks)
                    if feedback == "Good plank form! Hold it.":
                        if plank_start is None:
                            plank_start = time.time()
                        hold_time = int(time.time() - plank_start)
                        best_hold = max(best_hold, hold_time)
                    else:
                        plank_start = None
                        hold_time = 0

                    cv2.putText(frame, f"Feedback: {feedback}", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Hold: {hold_time}s", (50, 400), 1, 2, (0, 255, 0), 2)
                    cv2.putText(frame, f"Best: {best_hold}s", (50, 450), 1, 2, (255, 255, 0), 2)

        # 6. Inactivity reset
        if time.time() - last_detection_time > config.INACTIVE_THRESH:
            correct_count = 0
            incorrect_count = 0
            state_sequence = []
            plank_start = None
            hold_time = 0
            cv2.putText(frame, "RESETTING DUE TO INACTIVITY", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 7. UI overlay
        cv2.putText(frame, exercise_name, (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        if exercise in (1, 2, 3):
            cv2.putText(frame, f"Correct: {correct_count}", (50, 400), 1, 2, (0, 255, 0), 2)
            cv2.putText(frame, f"Incorrect: {incorrect_count}", (50, 450), 1, 2, (0, 0, 255), 2)

        cv2.imshow('KineSight AI Trainer', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    choice = show_menu()
    run_trainer(choice)
