import cv2
import time
import requests

params = cv2.SimpleBlobDetector_Params()

######### Change these params according to Need ############
params.filterByArea = True
params.minArea = 20
params.filterByCircularity = True
params.minCircularity = 0.5
params.filterByConvexity = False
params.filterByInertia = True
params.minInertiaRatio = 0.7
detector = cv2.SimpleBlobDetector_create(params)

width = 560
height = 480

##Set this after mounting the camera
left_circle = (100, 300)
right_circle = (500, 300)
# left_circle = (int(width/3), int(height/2))
# right_circle = (int(2*width/3), int(height/2))

esp32_ip = "192.168.1.100"
##################### FUNCTIONS #############################

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


def translate(x, y):
    print(x, y)

    translate_params = {
        "x": f"{x}",
        "y": f"{y}",
    }
    translate_url = f"http://{esp32_ip}/translate"

    translate_response = requests.get(translate_url, params=translate_params)
    print("Translate Response:")
    print(translate_response.text)


def rotate(direction):
    print(direction)
    rotate_params = {
    "dir": f"{direction}"
    }
    rotate_url = f"http://{esp32_ip}/rotate"

    rotate_response = requests.get(rotate_url, params=rotate_params)
    print("Rotate Response:")
    print(rotate_response.text)


def get_dist(vec1, vec2):
    return ((vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2)**1/2


def align_bot(centers):
    left_x = centers[0][0]
    left_y = centers[0][1]
    left_point = (left_x, left_y)
     
    right_x = centers[-1][0]
    right_y = centers[-1][1]
    right_point = (right_x, right_y)

    if left_y > right_y + 10 or left_y < right_y - 10:
        if left_y > right_y + 10:
            rotate('cw')
        elif left_y < right_y - 10:
            rotate('ccw')
    
    elif get_dist(left_point, left_circle) > 10:
        tx = (left_x - left_circle[0])
        ty = (left_y - left_circle[1])

        translate(tx, (ty*-1))



def main(cam):
    prev_time = time.time()
    frame_count = 0
    while True:
        ret, frame = cam.read()
        centers = []
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
            
            # img_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            try:
                for i in range(2):
                    center = (int(keypoints[i].pt[0]), int(keypoints[i].pt[1]))
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    centers.append(center)
                centers.sort(key= lambda point: point[0])
                # print(centers)
                align_bot(centers)
            except IndexError:
                    print("Not detected")

            cv2.circle(frame, left_circle, 5, (255, 0, 0), -1)
            cv2.circle(frame, right_circle, 5, (255, 0, 0), -1)

            cv2.imshow('Blob Detection', frame)
            # cv2.imshow('Blob Detection', img_with_keypoints)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # Uncomment if using jetson nano
    # cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    #for android
    cap = cv2.VideoCapture('http://192.168.29.186:4747/video')
    #laptop camera
    # cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    main(cap)