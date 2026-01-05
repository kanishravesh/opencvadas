import cv2
import mediapipe as mp


class FaceMeshTracker:
    """
    Face mesh tracker using MediaPipe.
    Responsible only for face detection and landmarks.
    """

    def __init__(self):
        self.face_mesh_module = mp.solutions.face_mesh

        self.face_mesh = self.face_mesh_module.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.drawer = mp.solutions.drawing_utils
        self.style = self.drawer.DrawingSpec(
            color=(0, 255, 0),
            thickness=1,
            circle_radius=1
        )
    
    def extract_landmarks(self, frame, results):
        """
        Extracts facial landmarks as pixel coordinates.
        Returns a list of (x, y) tuples.
        """
        frame_height, frame_width, _ = frame.shape
        landmark_points = []

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            for landmark in face_landmarks.landmark:
                x_pixel = int(landmark.x * frame_width)
                y_pixel = int(landmark.y * frame_height)
                landmark_points.append((x_pixel, y_pixel))

        return landmark_points

    
    
    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(rgb_frame)

    def draw(self, frame, results):
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.drawer.draw_landmarks(
                    frame,
                    face_landmarks,
                    self.face_mesh_module.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.style
                )
        return frame
