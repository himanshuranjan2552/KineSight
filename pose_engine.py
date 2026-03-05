import cv2
import mediapipe as mp

class PoseEngine:
    """
    Handles initialization and inference using MediaPipe Pose.
    Optimized for high-fidelity real-time body tracking on CPU.
    """
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Initialize MediaPipe Pose solution
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1, # 1 is balanced, 2 is higher accuracy, 0 is fastest
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def get_landmarks(self, frame):
        """
        Processes a frame and returns the frame with landmarks drawn 
        plus a dictionary of landmark coordinates.
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        landmarks_dict = None
        
        if results.pose_landmarks:
            # Draw skeleton on the frame for visualization
            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # Map landmarks to a dictionary for easy access by name
            landmarks_dict = {}
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = frame.shape
                # Store normalized coordinates (x, y, z) and visibility
                landmarks_dict[id] = {
                    'x': lm.x, 
                    'y': lm.y, 
                    'z': lm.z,
                    'vis': lm.visibility,
                    'px': int(lm.x * w), # pixel coordinate x
                    'py': int(lm.y * h)  # pixel coordinate y
                }
                
        return frame, landmarks_dict