import cv2
import numpy as np

def find_basketball_centers(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Skip every other frame to optimize performance
        if frame_count % 2 == 0:
            continue

        # Convert the frame to the HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the range of orange color in HSV
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([20, 255, 255])

        # Threshold the image to get only orange colors
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask=mask)

        # Convert the resulting image to grayscale
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        # Apply a blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use the HoughCircles function to detect circles in the frame
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30, param1=50, param2=20, minRadius=10, maxRadius=50
        )

        if circles is not None:
            # Convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

            # Draw all detected circles
            for (x, y, r) in circles:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                center = (x, y)
                cv2.circle(frame, center, 1, (0, 255, 0), 5)

            # Display the frame with the circles
            cv2.imshow("Basketball Detection", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(200) & 0xFF == ord("q"):  # 200 milliseconds delay (0.2 seconds)
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage:
video_path = "Videos/vid (1).mp4"
find_basketball_centers(video_path)
