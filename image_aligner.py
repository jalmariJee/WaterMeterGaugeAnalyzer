import cv2

import torch
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from For_testing_CNN.predict_callable import predict_digit
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "YOLO" / "best.pt"

# Get the directory of the current script
current_dir = os.path.dirname(__file__)
current_dir_images = os.path.join(current_dir, "images")

# Load the YOLO model
from ultralytics import YOLO
model = YOLO(MODEL_PATH)
model.eval()  # Set to evaluation mode (affects batch norm, dropout)

from PIL import Image
import torchvision.transforms as T

# Transform that matches your YOLO training transforms exactly
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=torch.tensor(0), std=torch.tensor(1))
])

# Create a text file to save the results
results_file_path = os.path.join(current_dir, "results.txt")
with open(results_file_path, "w") as results_file:
    results_file.write("Image Name, Predicted Digit, Confidence\n")  # Write header to the results file

# List all files in the images directory
image_files = [f for f in os.listdir(current_dir_images) if os.path.isfile(os.path.join(current_dir_images, f))]

#Comment out the loop for individual testing, using a single image for now to speed up development
for image in image_files:
    #Construct the relative path to the folder at the same level
    relative_path = os.path.join(current_dir_images, image)

    # Individual testing with a single image, using the "golden image" for development. Commented OUT
    #relative_path = os.path.join(current_dir_images, "IMG_20260309_205313.jpg") # IMG_20260309_205118_HDR.jpg" is the "golden image"

    # Read image
    img = cv2.imread(relative_path)

    if "Water" in image: # Check if the image name contains a specific substring
        #img = cv2.rotate(img, cv2.ROTATE_180)  # Rotate 180 degrees
        print(f"Rotated image: {image} by 180 degrees")
        waitKey = cv2.waitKey(0)  # Wait for a key press to proceed

    # If image is oriented incorrectly, rotate it
    if img.shape[1] > img.shape[0]:  # If width is greater than height
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # Rotate 90 degrees clockwise
        

    scale_percent = 100  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize the image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    resized_YOLO = resized.copy()
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
    gray2 = gray.copy()


    # Detect circles
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,         
        minDist=50,
        param1=100, # 200
        param2=85, # 100
        minRadius=40,      
        maxRadius=300
    )
    
    print(f"Detected circles: {circles}")  # Debugging statement to check detected circles
    # Order the circles based on their x-coordinate (left to right)
    if circles is not None:
        circles = circles[0, np.argsort(circles[0, :, 0])]  
        
    print(f"Ordered circles: {circles}")  # Debugging statement to check ordered circles



    # 3. Detection off small circles

    # Draw all detected circles
    if circles is not None:
        circles = np.round(circles).astype(int)
        count = 0
        for circle in circles:  # Process all the circles detected
            x, y, r = circle
            cv2.circle(gray, (x, y), r, (0, 255, 0), 2)  # Circle outline
            cv2.circle(gray, (x, y), 2, (0, 0, 255), 3)  # Center point
            

    if circles is not None:


        # Cropping all of the circles and predicting the digit in each of them, using the center point and radius for cropping
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        prediction_str = ""
        yolo_prediction_str = ""
        for circle in circles: 
            x, y, r = map(int, circle)

            
            # Square bounding box of size (2r x 2r) centered at (x, y)
            x1 = max(x - r, 0);        x2 = min(x + r, resized.shape[1])
            y1 = max(y - r, 0);        y2 = min(y + r, resized.shape[0])
            crop_img = resized[y1:y2, x1:x2]
            min_x = min(min_x, x1)
            min_y = min(min_y, y1)
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)


            # Predicting CNN
            prediction, confidence = predict_digit(crop_img)

            print(f"Predicted digit: {prediction} with confidence {confidence:.2f}% at point {(x, y)}")

            # Writin the predicted digit on the original image for visualization
            cv2.putText(resized, prediction, (x - r, y - r), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            # Also printing the confidence percentage on the image
            cv2.putText(resized, f"{confidence:.1f}%", (x - r, y - r + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            # Convert the cropped image to PIL format and apply the same transforms used in training
            crop_img_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            crop_img_tensor = transform(crop_img_pil).unsqueeze(0)  # Add batch dimension

            # Use direct model inference to avoid YOLO re-applying preprocessing
            with torch.no_grad():
                yolo_results = model(crop_img_tensor, verbose=False)
            
            # Extract prediction from Results object
            yolo_class_id = yolo_results[0].probs.top1
            yolo_confidence = yolo_results[0].probs.top1conf.item()
            yolo_prediction = yolo_results[0].names[yolo_class_id]
            
            # DEBUG: Show all top 5 predictions
            top5_conf = yolo_results[0].probs.top5conf
            top5_classes = yolo_results[0].probs.top5
            print(f"  YOLO Top 5: {[(yolo_results[0].names[int(c)], conf.item()) for c, conf in zip(top5_classes, top5_conf)]}")
            print(f"  → CNN: {prediction} ({confidence:.1f}%)  |  YOLO: {yolo_prediction} ({yolo_confidence*100:.1f}%)")
            
            cv2.putText(resized_YOLO,  str(yolo_prediction), (x - r, y - r), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            cv2.putText(resized_YOLO, f"{yolo_confidence*100:.1f}%", (x - r, y - r + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            # Comment the cropping, using these circles for alignment
            #Save the cropped image

            # The results consists of 4 readings: lets save them all in string
            prediction_str = prediction_str +  prediction
            yolo_prediction_str = yolo_prediction_str + str((yolo_prediction.replace("digit_", "")))


        dial_crop = resized[min_y-30:max_y+30, min_x-30:max_x+30]
        dial_crop_YOLO = resized_YOLO[min_y-30:max_y+30, min_x-30:max_x+30]

        output_path = os.path.join(current_dir, "processed_images_CNN")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # Writing the image name with the predicted digit for easier identification of the results
        # If the exact same image name already exists, create a new name with a suffix to avoid overwriting
        suffix = ""
        if os.path.exists(os.path.join(output_path, f"{prediction_str}.jpg")):
            suffix = 1
            
            while os.path.exists(os.path.join(output_path, f"{prediction_str}_{suffix}.jpg")):
                suffix += 1

            cv2.imwrite(os.path.join(output_path, f"{prediction_str}_{suffix}.jpg"), dial_crop)
        else:
            cv2.imwrite(os.path.join(output_path, f"{prediction_str}.jpg"), dial_crop)
        with open(results_file_path, "a") as results_file:
            results_file.write(f"{prediction_str}_{suffix}, {prediction_str}, {confidence:.2f}%\n")

        # cv2.imwrite(os.path.join(output_path, f"{predict ion}"), dial_crop)
        # cv2.imwrite(os.path.join(output_path, f"processed_{image}"), dial_crop)

        output_path = os.path.join(current_dir, "processed_images_YOLO")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if os.path.exists(os.path.join(output_path, f"{yolo_prediction_str}.jpg")):
            suffix = 1
            while os.path.exists(os.path.join(output_path, f"{yolo_prediction_str}_{suffix}.jpg")):
                suffix += 1
            cv2.imwrite(os.path.join(output_path, f"{yolo_prediction_str}_{suffix}.jpg"), dial_crop_YOLO)
        else:
            cv2.imwrite(os.path.join(output_path, f"{yolo_prediction_str}.jpg"), dial_crop_YOLO)

        # cv2.imwrite(os.path.join(output_path, f"{prediction}"), dial_crop)
        #cv2.imwrite(os.path.join(output_path, f"processed_{image}"), dial_crop_YOLO)



    ##Debugging and checking coordinates
    """
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at: ({x}, {y})")
    cv2.namedWindow("Detected Circles")
    cv2.setMouseCallback("Detected Circles", click_event)

    """

    #cv2.imshow('Detected Circles', resized)
    #plt.title('Detected Circles')
    #cv2.waitKey(0)




    # Show result
    #cv2.imshow('Detected Circle', crop_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

