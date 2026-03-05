import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os

class PoseEngine:
    """
    Handles initialization and inference using MediaPipe Pose.
    Optimized for high-fidelity real-time body tracking on CPU.
    Updated for MediaPipe 0.10.32+ API.
    """
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Download pose landmarker model if not exists
        model_path = self._get_model_path()
        
        # Initialize MediaPipe Pose with new tasks API
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        # Create pose landmarker options
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        self.frame_counter = 0
        
        # Pose connections for drawing
        self.POSE_CONNECTIONS = frozenset([
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
            (27, 29), (28, 30), (27, 31), (28, 32), (29, 31), (30, 32)
        ])
    
    def _get_model_path(self):
        """Download the pose landmarker model if it doesn't exist."""
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'pose_landmarker_lite.task')
        
        if not os.path.exists(model_path):
            print("Downloading pose landmarker model...")
            url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded successfully!")
        
        return model_path

    def get_landmarks(self, frame):
        """
        Processes a frame and returns the frame with landmarks drawn 
        plus a dictionary of landmark coordinates.
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Process the frame
        self.frame_counter += 1
        timestamp_ms = self.frame_counter * 33  # Approximate 30fps
        results = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        landmarks_dict = None
        
        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            pose_landmarks = results.pose_landmarks[0]
            
            # Draw skeleton on the frame
            self._draw_landmarks(frame, pose_landmarks)
            
            # Map landmarks to a dictionary for easy access by name
            landmarks_dict = {}
            h, w, c = frame.shape
            for id, lm in enumerate(pose_landmarks):
                landmarks_dict[id] = {
                    'x': lm.x, 
                    'y': lm.y, 
                    'z': lm.z,
                    'vis': lm.visibility,
                    'px': int(lm.x * w),  # pixel coordinate x
                    'py': int(lm.y * h)   # pixel coordinate y
                }
                
        return frame, landmarks_dict
    
    def _draw_landmarks(self, frame, landmarks):
        """Draw pose landmarks and connections on the frame."""
        h, w, c = frame.shape
        
        # Draw connections
        for connection in self.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)