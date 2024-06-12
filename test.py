import cv2
import numpy as np

img = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)

params = cv2.SimpleBlobDetector_Params()

params.filterByArea = False
params.minArea = 1
params.filterByCircularity = False
params.minCircularity = 0.5
params.filterByConvexity = False
params.filterByInertia = True
params.minInertiaRatio = 0.4

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(img)
# img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

for keypoint in keypoints:
    center = (int(keypoint.pt[0]), int(keypoint.pt[1]))
    cv2.circle(img, center, 5, (0, 0, 255), -1)

cv2.imshow('Blob Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
