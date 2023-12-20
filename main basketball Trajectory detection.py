import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_basketball_trajectory(video_path, polynomial_degree=2):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    actual_trajectory = []
    predicted_trajectory = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

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

            # Find the circle with the desired color
            for (x, y, r) in circles:
                if mask[y, x] == 255:
                    # Draw the circle on the frame
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                    center = (x, y)
                    cv2.circle(frame, center, 1, (0, 255, 0), 5)

                    # Store the center coordinates in the actual trajectory list
                    actual_trajectory.append(center)

                    # Perform polynomial regression on the actual trajectory
                    if len(actual_trajectory) >= polynomial_degree + 1:
                        coefficients = np.polyfit(
                            np.arange(len(actual_trajectory)),
                            np.array(actual_trajectory)[:, 0],
                            polynomial_degree
                        )
                        predicted_x = np.polyval(coefficients, np.arange(len(actual_trajectory)))
                        predicted_y = np.array(actual_trajectory)[:, 1]

                        # Combine the predicted x and actual y to form the predicted trajectory
                        predicted_trajectory = list(zip(predicted_x, predicted_y))

                        # Draw a line connecting consecutive points in the actual trajectory
                        if len(actual_trajectory) > 1:
                            cv2.line(frame, actual_trajectory[-2], actual_trajectory[-1], (255, 0, 0), 2)

                        # Draw a line connecting consecutive points in the predicted trajectory
                        if len(predicted_trajectory) > 1:
                            pt1 = (int(predicted_trajectory[-2][0]), int(predicted_trajectory[-2][1]))
                            pt2 = (int(predicted_trajectory[-1][0]), int(predicted_trajectory[-1][1]))
                            cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

            # Display the frame with the circle and trajectories
            cv2.imshow("Basketball Detection", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(200) & 0xFF == ord("q"):  # 200 milliseconds delay (0.2 seconds)
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

    # Convert the trajectory lists to NumPy arrays for easier manipulation
    actual_trajectory = np.array(actual_trajectory)

    # Plot the actual and predicted trajectories
    plt.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], marker='o', linestyle='-', color='b', label='Actual Trajectory')
    if len(predicted_trajectory) > 0:
        plt.plot(np.array(predicted_trajectory)[:, 0], np.array(predicted_trajectory)[:, 1], marker='o', linestyle='-', color='r', label='Predicted Trajectory')
    plt.title('Actual vs Predicted Basketball Trajectory')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.legend()
    plt.show()

# Example usage:
video_path = "Videos/vid (1).mp4"
find_basketball_trajectory(video_path)
