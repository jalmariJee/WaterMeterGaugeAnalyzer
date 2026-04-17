from ultralytics import YOLO
import cv2

print("🧠 Loading custom YOLO brain...")
# Load the model you just downloaded from Colab
model = YOLO('best.pt')

# Pick ONE image from your dataset to test
# (Change this path if your image is named differently)
test_image = "yolo_dataset/images/WaterMeter_64.jpg"

print(f"👀 Looking at {test_image}...")
# Run inference (ask the AI to find the dials)
results = model(test_image) 

# Draw the boxes and show the image on the screen
for r in results:
    # r.plot() creates a new image with the colorful boxes and labels drawn on it
    annotated_image = r.plot() 
    
    cv2.imshow("YOLOv8 Vision", annotated_image)
    print("✅ Press any key on your keyboard to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()