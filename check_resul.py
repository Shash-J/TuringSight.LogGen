import cv2

rtsp_url = "rtsp://admin:Tech@007@106.51.57.8:554/cam/realmonitor?channel=1&subtype=0"

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Failed to open stream")
else:
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Width: {width}")
    print(f"Height: {height}")
    print(f"FPS: {fps}")

cap.release()