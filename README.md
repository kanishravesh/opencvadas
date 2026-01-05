This project is a real-time computer vision application built using MediaPipe and OpenCV. It processes live video input from a webcam and detects visual landmarks and orientations in real time. The project was developed to understand how modern computer vision pipelines operate when working with continuous video streams rather than static images.

The application runs entirely on a local machine and performs inference on the CPU.

Overview

The system captures frames from a webcam and processes them using a MediaPipe solution. MediaPipe detects facial or pose landmarks and computes orientation-related values that describe how a subject’s head or body is positioned in three-dimensional space. OpenCV is used to handle video capture, frame conversion, and visualization.

This project focuses on real-time processing, latency awareness, and understanding how landmark-based computer vision models behave under different conditions such as movement, lighting, and camera angle.

Head orientation and angle estimation

As part of this project, head orientation values such as pitch, yaw, and roll were computed and observed.

Pitch refers to the up-and-down movement of the head. It describes whether the head is tilted upward or downward. In practical terms, pitch changes when a person looks up toward the ceiling or down toward the ground.

Yaw refers to the left-and-right rotation of the head. It represents horizontal turning, such as when a person looks to the left or right side. Yaw is commonly used to understand attention direction or gaze orientation.

Roll refers to the sideways tilt of the head. It describes rotation along the axis that runs from the nose to the back of the head. Roll changes when a person tilts their head toward their shoulder.

These three angles together describe the orientation of the head in 3D space and are widely used in computer vision tasks such as driver monitoring systems, attention tracking, and human-computer interaction.

How the system works

The application continuously reads frames from a webcam using OpenCV. Each frame is converted into the format required by MediaPipe and passed through a pre-trained model. MediaPipe detects landmarks and calculates orientation-related values based on their spatial relationships.

The detected landmarks and orientation information are then rendered back onto the video stream. This allows the user to visually observe changes in pitch, yaw, and roll as they move their head in front of the camera.

The processing loop is designed to be efficient enough for real-time feedback on typical hardware.

Why MediaPipe was chosen

MediaPipe provides optimized and reliable computer vision pipelines that work well in real-time environments without requiring custom model training. It abstracts the complexity of deep learning inference while still exposing meaningful outputs such as landmark coordinates and orientation metrics.

Using MediaPipe allowed the project to focus on understanding system integration, real-time constraints, and interpretation of model outputs rather than building models from scratch.

Performance characteristics

The application attempts to maintain interactive frame rates while processing video input. Performance depends on factors such as camera resolution, lighting conditions, and available CPU resources. On most standard systems, the application runs smoothly with minimal delay.

Because the system relies on CPU-based inference, performance may decrease on lower-end machines or under heavy system load.

Limitations

The accuracy of landmark detection and angle estimation is sensitive to lighting conditions, camera quality, and face visibility. Occlusions, fast movements, or extreme head rotations can reduce stability.
