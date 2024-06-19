import cv2
import cv2.cuda as cuda
import numpy as np
import time

def main():
  # Initialize CUDA context (if using CUDA)
  if cuda.getCudaEnabledDeviceCount() > 0:
    cuda.init()

  # Open video capture
  cap = cv2.VideoCapture(0)  # Replace 0 with video file path if needed

  # Blob detection parameters
  params = cv2.SimpleBlobDetector_Params()
  params.filterByArea = True
  params.minArea = 20
  params.filterByCircularity = True
  params.minCircularity = 0.5
  params.filterByConvexity = False
  params.filterByInertia = True
  params.minInertiaRatio = 0.7

  # Create blob detector (on GPU or CPU)
  if cuda.getCudaEnabledDeviceCount() > 0:
    print('on gpu0')
    gpu_params = cuda.SimpleBlobDetector_Params()
    gpu_params.filterByArea = True
    gpu_params.minArea = 20
    gpu_params.filterByCircularity = True
    gpu_params.minCircularity = 0.5
    gpu_params.filterByConvexity = False
    gpu_params.filterByInertia = True
    gpu_params.minInertiaRatio = 0.7
    detector = cuda.createSimpleBlobDetector(gpu_params)
  else:
    detector = cv2.SimpleBlobDetector_create(params)

  prev_time = time.time()
  frame_count = 0
  while True:
    # Capture frame
    ret, frame = cap.read()

    if not ret:
      print("Error: Could not capture frame from video.")
      break

    # Process frame on GPU (if available)
    if cuda.getCudaEnabledDeviceCount() > 0:
      print('on gpu1')
      gpu_frame = cuda.GpuMat()
      gpu_frame.upload(frame)
      gpu_keypoints = detector.detect(gpu_frame)
      keypoints = gpu_keypoints.download()
    else:
      # Process frame on CPU
      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
      keypoints = detector.detect(gray_frame)

    # Calculate frame rate
    curr_time = time.time()
    if curr_time - prev_time >= 1:
      frame_rate = frame_count / (curr_time - prev_time)
      print(f"Frame Rate: {frame_rate:.2f} fps")
      prev_time = curr_time

    # Draw keypoints on image
    img_with_keypoints = cv2.drawKeypoints(frame.copy(), keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display image
    cv2.imshow('Blob Detection (GPU)', img_with_keypoints)
    frame += 1

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Clean up
  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()