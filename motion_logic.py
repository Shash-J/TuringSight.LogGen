import cv2

class MotionDetector:
    def __init__(self, threshold=25, min_area=5000):
        self.threshold = threshold
        self.min_area = min_area
        self.avg_frame = None

    def has_motion(self, frame):
        # 1. Pre-process frame (Grayscale and Blur)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # 2. Initialize background model
        if self.avg_frame is None:
            self.avg_frame = gray.copy().astype("float")
            return False

        # 3. Accumulate weighted average to handle lighting changes
        cv2.accumulateWeighted(gray, self.avg_frame, 0.5)
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg_frame))

        # 4. Thresholding
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # 5. Check if the "size" of motion is large enough
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                return True # Significant motion found
        
        return False