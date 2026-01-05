import cv2
import numpy as np
import math


class HeadPoseEstimator:
    """
    Estimates head pose (yaw, pitch, roll) using facial landmarks.
    """

    def __init__(self):

        self.face_model_points = np.array([
            (0.0, 0.0, 0.0),        # Nose tip
            (0.0, -63.6, -12.5),   # Chin
            (-43.3, 32.7, -26.0),  # Left eye corner
            (43.3, 32.7, -26.0),   # Right eye corner
            (-28.9, -28.9, -24.1), # Left mouth corner
            (28.9, -28.9, -24.1)   # Right mouth corner
        ])

        self.landmark_indices = [
            1,    # Nose tip
            152,  # Chin
            33,   # Left eye
            263,  # Right eye
            61,   # Left mouth
            291   # Right mouth
        ]

    def estimate_pose(self, frame, landmarks):
        """
        Returns:
            yaw, pitch, roll (in degrees)
        """
        image_height, image_width, _ = frame.shape

        image_points = np.array(
            [landmarks[i] for i in self.landmark_indices],
            dtype="double"
        )

        focal_length = image_width
        center = (image_width / 2, image_height / 2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        distortion_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.face_model_points,
            image_points,
            camera_matrix,
            distortion_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None, None, None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

        pitch = angles[0] * 360
        yaw = angles[1] * 360
        roll = angles[2] * 360

        return yaw, pitch, roll
