import cv2
import numpy as np

def find_basketball_center(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply a blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use the HoughCircles function to detect circles in the frame
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=50
        )

        if circles is not None:
            # Convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

            # Draw the circle on the frame
            for (x, y, r) in circles:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                center = (x, y)
                cv2.circle(frame, center, 1, (0, 255, 0), 5)

            # Display the frame with the circle
            cv2.imshow("Basketball Detection", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage:
video_path = "Videos/vid (1).mp4"
find_basketball_center(video_path)
