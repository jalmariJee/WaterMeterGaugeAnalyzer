from ultralytics import YOLO

# 1. Load the "Empty Shell"
# 'yolov8n.pt' stands for "Nano". It is the fastest, lightest version of YOLO.
# It automatically downloads the pre-trained weights to start.
model = YOLO('yolov8n.pt') 

# 2. Train the Model
print("🚀 Starting YOLO Training...")
results = model.train(
    data='data.yaml',   # THIS IS CRITICAL: The map to your images and bounding box text files
    epochs=10,          # How many times to loop through the dataset
    imgsz=640,          # YOLO automatically squishes your raw photos into 640x640 squares
    device='0',       # For GPU purpose
    plots=True          # Automatically draws beautiful charts of your accuracy!
)
print("✅ Training Complete!")