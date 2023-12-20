import math
import cv2
import numpy as np
import cvzone
from cvzone.ColorModule import ColorFinder
import tkinter as tk
from tkinter import simpledialog, filedialog

# Function to get video speed from the user using a GUI window
def get_video_speed():
    root = tk.Tk()
    root.withdraw()
    speed = simpledialog.askfloat("Video Speed", "Enter video speed (e.g., 0.5 for half speed):", minvalue=0.1, maxvalue=2.0)
    return speed

# Function to get the video file path from the user using a file dialog
def get_video_path():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi")])
    return file_path

# Get video speed from the user
video_speed = get_video_speed()

# Get the video file path from the user or use the default path
video_path = get_video_path()
if not video_path:
    video_path = 'Videos/vid (4).mp4'

# Initialize the Video
cap = cv2.VideoCapture(video_path)

# Set the video speed
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / (fps * video_speed))

# Variables
posListX, posListY = [], []
xList = [item for item in range(0, 1300)]
prediction = False

def find_basketball_center(frame):
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of the orange color in HSV
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
                return (x, y)

    return None

# Set constant values for hoop position
hoop_position_constant = (500, 100)

while True:
    # Grab the image
    success, img = cap.read()

    if not success:
        break

    img = img[0:900, :]

    # Find the Basketball Center
    basketball_center = find_basketball_center(img)

    # Use the constant hoop position
    hoop_position = hoop_position_constant

    if basketball_center is not None and hoop_position is not None:
        posListX.append(basketball_center[0])
        posListY.append(basketball_center[1])

        # Polynomial Regression y = Ax^2 + Bx + C
        # Find the Coefficients
        A, B, C = np.polyfit(posListX, posListY, 2)

        for i, (posX, posY) in enumerate(zip(posListX, posListY)):
            pos = (posX, posY)
            cv2.circle(img, pos, 10, (0, 255, 0), cv2.FILLED)
            if i == 0:
                cv2.line(img, pos, pos, (0, 255, 0), 5)
            else:
                cv2.line(img, pos, (posListX[i - 1], posListY[i - 1]), (0, 255, 0), 5)

        for x in xList:
            y = int(A * x ** 2 + B * x + C)
            cv2.circle(img, (x, y), 2, (255, 0, 255), cv2.FILLED)

        # Draw the trajectory line outside the loop after the initial prediction
        if len(posListX) >= 2:
            for x in range(len(posListX) - 1):
                cv2.line(img, (posListX[x], posListY[x]), (posListX[x + 1], posListY[x + 1]), (0, 255, 0), 2)

        if len(posListX) < 10:
            # Prediction
            # X values 330 to 430  Y 590
            a = A
            b = B
            c = C - hoop_position[1]

            if not math.isnan(-b - np.sqrt(b ** 2 - (4 * a * c))) / (2 * a):
                x = int((-b - np.sqrt(b ** 2 - (4 * a * c))) / (2 * a))
            prediction = 330 < x < 430

    # Display
    img = cv2.resize(img, (0, 0), None, 0.7, 0.7)

    # Draw the text outside the loop after the initial prediction
    if prediction:
        cvzone.putTextRect(img, "Basket", (50, 150),
                           scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
    else:
        cvzone.putTextRect(img, "No Basket", (50, 150),
                           scale=5, thickness=5, colorR=(0, 0, 200), offset=20)

    cv2.imshow("Image", img)

    # Check if the space bar is pressed
    key = cv2.waitKey(delay) & 0xFF
    if key == 27:  # Press 'Esc' to exit
        break

cv2.destroyAllWindows()
cap.release()