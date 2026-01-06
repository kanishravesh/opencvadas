import cv2
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from face_utils import FaceMeshTracker
from drowsiness import DrowsinessDetector
from head_pose import HeadPoseEstimator
from inference import DriverStateAnalyzer


st.set_page_config(
    page_title="Driver Monitoring System",
    layout="wide"
)

st.title("Driver Monitoring System")
st.write(
    "Real-time driver monitoring using facial landmarks, "
    "drowsiness detection, yawning detection, and head pose estimation."
)



face_tracker = FaceMeshTracker()
drowsiness_detector = DrowsinessDetector()
head_pose_estimator = HeadPoseEstimator()
state_analyzer = DriverStateAnalyzer()


class DriverVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = face_tracker.detect(img)
        img = face_tracker.draw(img, results)

        landmarks = face_tracker.extract_landmarks(img, results)
        driver_state = "NO FACE DETECTED"

        if landmarks:
            ear, is_drowsy = drowsiness_detector.check_drowsiness(landmarks)
            mar, is_yawning = drowsiness_detector.check_yawning(landmarks)
            yaw, pitch, _ = head_pose_estimator.estimate_pose(img, landmarks)

            driver_state = state_analyzer.analyze(
                is_drowsy=is_drowsy,
                is_yawning=is_yawning,
                yaw=yaw,
                pitch=pitch
            )

            cv2.putText(
                img,
                f"EAR: {ear:.2f}  MAR: {mar:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            if yaw is not None:
                cv2.putText(
                    img,
                    f"Yaw: {yaw:.1f}  Pitch: {pitch:.1f}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

        cv2.putText(
            img,
            f"STATE: {driver_state}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255) if driver_state != "ALERT" else (0, 255, 0),
            3
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="driver-monitoring",
    video_processor_factory=DriverVideoProcessor,
    media_stream_constraints={
        "video": {
            "width": 320,
            "height": 240,
            "frameRate": 15
        },
        "audio": False
    },
    async_processing=False, 
)


