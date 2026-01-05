class DriverStateAnalyzer:
    """
    Combines drowsiness, yawning, and head pose
    to determine the driver's current state.
    """

    def __init__(self):
        self.distraction_frame_count = 0
        self.distraction_frame_limit = 20

        self.yaw_limit = 25      # degrees
        self.pitch_limit = 20    # degrees

    def analyze(
        self,
        is_drowsy,
        is_yawning,
        yaw,
        pitch
    ):
        """
        Returns a human-readable driver state.
        """

        # Highest priority: drowsiness
        if is_drowsy:
            return "DROWSY - TAKE A BREAK"

        # Fatigue warning
        if is_yawning:
            return "YAWNING - FATIGUE DETECTED"

        # Distraction logic
        distracted = False
        if yaw is not None and pitch is not None:
            if abs(yaw) > self.yaw_limit or pitch > self.pitch_limit:
                self.distraction_frame_count += 1
            else:
                self.distraction_frame_count = 0

            distracted = (
                self.distraction_frame_count >= self.distraction_frame_limit
            )

        if distracted:
            return "DISTRACTED - WATCH THE ROAD"

        return "ALERT"
