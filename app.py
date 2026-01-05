import cv2
import gradio as gr


from face_utils import FaceMeshTracker
from drowsiness import DrowsinessDetector
from head_pose import HeadPoseEstimator
from inference import DriverStateAnalyzer


face_tracker = FaceMeshTracker()
drowsiness_detector = DrowsinessDetector()
head_pose_estimator = HeadPoseEstimator()
state_analyzer = DriverStateAnalyzer()



def analyze_driver(video):
    cap = cv2.VideoCapture(video)

    final_frame = None
    final_state = "NO DATA"

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = face_tracker.detect(frame)
        frame = face_tracker.draw(frame, results)

        landmarks = face_tracker.extract_landmarks(frame, results)

        driver_state = "NO FACE DETECTED"

        if landmarks:
            ear, is_drowsy = drowsiness_detector.check_drowsiness(landmarks)
            mar, is_yawning = drowsiness_detector.check_yawning(landmarks)
            yaw, pitch, _ = head_pose_estimator.estimate_pose(frame, landmarks)

            driver_state = state_analyzer.analyze(
                is_drowsy=is_drowsy,
                is_yawning=is_yawning,
                yaw=yaw,
                pitch=pitch
            )

        final_frame = frame
        final_state = driver_state

    cap.release()

    final_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
    return final_frame, final_state



# Gradio Interface
demo = gr.Interface(
    fn=analyze_driver,
    inputs=gr.Video(source="webcam"),
    outputs=[
        gr.Image(label="Driver Monitoring View"),
        gr.Textbox(label="Driver State")
    ],
    title="Driver Monitoring System (DMS)",
    description="Driver monitoring using face landmarks, EAR, MAR and head pose."
)


demo.launch()
