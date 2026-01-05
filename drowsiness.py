import numpy as np
import math


class DrowsinessDetector:
    """
    Detects driver drowsiness using Eye Aspect Ratio (EAR).
    """

    def __init__(
        self,
        eye_closed_threshold=0.20,
        consecutive_frames_limit=15
    ):
        self.eye_closed_threshold = eye_closed_threshold
        self.consecutive_frames_limit = consecutive_frames_limit
        self.closed_eye_frame_count = 0


        self.left_eye_points = [33, 160, 158, 133, 153, 144]
        self.right_eye_points = [362, 385, 387, 263, 373, 380]

        self.mouth_points = [
            61, 81, 13, 311, 308, 402, 14, 178
        ]

        self.yawn_threshold = 0.6
        self.yawn_frame_count = 0
        self.yawn_frame_limit = 10


    def _distance(self, point1, point2):
        return math.dist(point1, point2)

    def _eye_aspect_ratio(self, eye_points):
        vertical_1 = self._distance(eye_points[1], eye_points[5])
        vertical_2 = self._distance(eye_points[2], eye_points[4])
        horizontal = self._distance(eye_points[0], eye_points[3])

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def _mouth_aspect_ratio(self, mouth_points):
        vertical_1 = self._distance(mouth_points[1], mouth_points[7])
        vertical_2 = self._distance(mouth_points[2], mouth_points[6])
        horizontal = self._distance(mouth_points[0], mouth_points[4])

        mar = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return mar
    

    def check_drowsiness(self, landmarks):
        """
        Returns:
            ear_value (float)
            is_drowsy (bool)
        """

        left_eye = [landmarks[i] for i in self.left_eye_points]
        right_eye = [landmarks[i] for i in self.right_eye_points]

        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)

        average_ear = (left_ear + right_ear) / 2.0

        if average_ear < self.eye_closed_threshold:
            self.closed_eye_frame_count += 1
        else:
            self.closed_eye_frame_count = 0

        is_drowsy = self.closed_eye_frame_count >= self.consecutive_frames_limit

        return average_ear, is_drowsy


    def check_yawning(self, landmarks):
        """
        Returns:
            mar_value (float)
            is_yawning (bool)
        """

        mouth = [landmarks[i] for i in self.mouth_points]
        mar = self._mouth_aspect_ratio(mouth)

        if mar > self.yawn_threshold:
            self.yawn_frame_count += 1
        else:
            self.yawn_frame_count = 0

        is_yawning = self.yawn_frame_count >= self.yawn_frame_limit
        return mar, is_yawning
