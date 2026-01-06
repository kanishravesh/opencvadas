import cv2
from face_utils import FaceMeshTracker
from drowsiness import DrowsinessDetector
from head_pose import HeadPoseEstimator
from inference import DriverStateAnalyzer


camera = cv2.VideoCapture(0)
head_pose_estimator = HeadPoseEstimator()
state_analyzer = DriverStateAnalyzer()

face_tracker = FaceMeshTracker()
drowsiness_detector = DrowsinessDetector()

while True:
    success, frame = camera.read()
    if not success:
        break

    results = face_tracker.detect(frame)
    frame = face_tracker.draw(frame, results)

    landmarks = face_tracker.extract_landmarks(frame, results)

    if landmarks:
        ear, is_drowsy = drowsiness_detector.check_drowsiness(landmarks)
        mar, is_yawning = drowsiness_detector.check_yawning(landmarks)
        yaw, pitch, roll = head_pose_estimator.estimate_pose(frame, landmarks)




        status_text = f"EAR: {ear:.2f} | MAR: {mar:.2f}"
        if yaw is not None:
           pose_text = f"Yaw: {yaw:.1f}  Pitch: {pitch:.1f}"
           cv2.putText(
           frame,
           pose_text,
          (20, 80),
           cv2.FONT_HERSHEY_SIMPLEX,
          0.9,
          (255, 255, 0),
          2
     )    
    

        if is_yawning:
            status_text += "  |  YAWNING"
        if is_drowsy:
            status_text += "  |  DROWSY!"

        cv2.putText(
            frame,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255) if (is_drowsy or is_yawning) else (0, 255, 0),
            2
        )
        
    driver_state = state_analyzer.analyze(    
                                is_drowsy=is_drowsy,
                                is_yawning=is_yawning,
                                yaw=yaw,
                                pitch=pitch
    )
    
    cv2.putText(
        frame,
        f"Driver State: {driver_state}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255) if driver_state != "ALERT" else (0, 255, 0),
        3
        )

    cv2.imshow("Driver Monitoring - Drowsiness", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()



