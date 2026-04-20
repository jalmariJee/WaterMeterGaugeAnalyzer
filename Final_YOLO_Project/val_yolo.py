from ultralytics import YOLO

# 1. Loading final trained brain
model = YOLO('final_water_meter_model.pt')

# 2. Run the official validation suite
print("🚀 Starting Accuracy Analysis...")
metrics = model.val(
    data='yolo_dataset/data.yaml', 
    split='val', # Tells YOLO to look at the validation set
    plots=True   # This is the trigger for the Confusion Matrix!
)

print("✅ Analysis Complete! Check the 'runs/detect/val' folder for your Confusion Matrix.")