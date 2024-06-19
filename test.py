import cv2
import numpy as np
import time


# img = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 20
params.filterByCircularity = True
params.minCircularity = 0.5
params.filterByConvexity = False
params.filterByInertia = True
params.minInertiaRatio = 0.7
detector = cv2.SimpleBlobDetector_create(params)

prev_time = time.time()
frame_count = 0

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()

    if ret:
        curr_time = time.time()
        if curr_time - prev_time >= 1:  # Calculate frame rate every 1 second
            frame_rate = frame_count / (curr_time - prev_time)
            print(f"Frame Rate: {frame_rate:.2f} fps")
            prev_time = curr_time
            frame_count = 0
        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints = detector.detect(gray)
        img_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # for keypoint in keypoints:
        #     center = (int(keypoint.pt[0]), int(keypoint.pt[1]))
        #     cv2.circle(frame, center, 5, (0, 0, 255), -1)

        # cv2.imshow('Blob Detection', frame)
        cv2.imshow('Blob Detection', img_with_keypoints)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
