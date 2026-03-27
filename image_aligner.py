import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

current_dir_images = os.path.join(current_dir, "images")

# List all files in the images directory
image_files = [f for f in os.listdir(current_dir_images) if os.path.isfile(os.path.join(current_dir_images, f))]

for image in image_files:
    # Skip files that aren't WaterMeter_... images ---
    if not image.startswith("WaterMeter_"):
        continue
    relative_path = os.path.join(current_dir_images, image)
    #Construct the relative path to the folder at the same level
    #relative_path = os.path.join(current_dir_images, "IMG_20260309_203637_HDR.jpg")

    # Read image
    img = cv2.imread(relative_path)


    # If image is oriented incorrectly, rotate it
    if img.shape[1] > img.shape[0]:  # If width is greater than height
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # Rotate 90 degrees clockwise

    scale_percent = 100  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize the image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    #hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # lower and upper bounds for red color
    #lower_red = np.array([0,50,50]) 
    #upper_red = np.array([10,255,255])

    # Filter red color from HSV image
    #mask = cv2.inRange(hsv, lower_red, upper_red)
    #result = cv2.bitwise_and(resized, resized, mask=mask)

    # resized = img.copy()
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # gray = resized(0:height, 0:width, 0)

    #plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)) # Display with matplotlib
    #plt.title('Red Channel Isolated Image')
    #plt.show()
    # cv2.imshow('Detected Circle', gray)

    # Reduce noise
    gray = cv2.medianBlur(gray, 5)



    # Detect circles
    # Lowered maxRadius so it finds dials, not the whole meter ---
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,         
        minDist=50,
        param1=200,
        param2=80,
        minRadius=40,      
        maxRadius=120
    )

    print(circles)

    # Draw all detected circles
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        count = 0
        for circle in circles:  # Process all the circles detected
            x, y, r = circle
            cv2.circle(gray, (x, y), r, (0, 255, 0), 2)  # Circle outline
            cv2.circle(gray, (x, y), 2, (0, 0, 255), 3)  # Center point
            

    if circles is not None:
        # circles is an array/list of [x, y, r]
        sorted_circles = sorted(circles, key=lambda c: c[0])  # sort by x (column)
        x, y, r = map(int, sorted_circles[0])

        # Sort the leftmost circle and crop the image around it
        # -- Added padding and fixed the save path to overwrite ---
        padding = 10 # This gives the dial a little breathing room so edges aren't cut off

        # Square bounding box of size (2r x 2r) centered at (x, y)
        x1 = max(x - r - padding, 0);        x2 = min(x + r + padding, resized.shape[1])
        y1 = max(y - r - padding, 0);        y2 = min(y + r + padding, resized.shape[0])
        crop_img = resized[y1:y2, x1:x2]

        #Save the cropped image
        output_path = os.path.join(current_dir, "cropped_images")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        cv2.imwrite(os.path.join(output_path, image), crop_img)

        


        # Show result
        #cv2.imshow('Detected Circle', crop_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

