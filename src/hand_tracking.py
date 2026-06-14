import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as MPImage, ImageFormat

class HandTracker:
    def __init__(self, static_image_mode=False, max_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Create HandDetector options
        base_options = python.BaseOptions(model_asset_path=None)
        options = vision.HandDetectorOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence
        )
        self.detector = vision.HandDetector.create_from_options(options)
        self.results = None

    def find_hands(self, frame, draw=True):
        # Convert frame to mediapipe Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=frame_rgb)
        
        # Detect hands
        self.results = self.detector.detect(mp_image)
        
        # Draw landmarks if requested
        if draw and self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        return frame

    def get_landmarks(self, frame):
        landmarks_list = []
        if self.results and self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                for landmark in hand_landmarks:
                    landmarks_list.extend([landmark.x, landmark.y, landmark.z])
        return landmarks_list