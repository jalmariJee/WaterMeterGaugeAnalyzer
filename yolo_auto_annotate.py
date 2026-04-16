from ultralytics import YOLO
import os

# --- Configuration ---
RAW_IMAGES_DIR = "yolo_dataset/images"
YOLO_LABELS_DIR = "yolo_dataset/labels"
MODEL_PATH = "best.pt"  # The 150-epoch YOLO brain, refers only to the first 70 images, annotated and corrected.

# Load your YOLO model
print("🧠 Loading YOLOv8 Object Detector...")
model = YOLO(MODEL_PATH)

print(f"🚀 Starting YOLO-assisted labeling...")
image_files = sorted([f for f in os.listdir(RAW_IMAGES_DIR) if f.lower().endswith('.jpg')])

for filename in image_files:
    # We only want to auto-annotate the files that haven't done manually; i.e. images (71-576)
    # Assuming the files are named strictly like "WaterMeter_71.jpg"
    try:
        # Extract the number from the filename (e.g., "71")
        img_num = int(filename.split('_')[1].split('.')[0])
        
        # SKIP the first 70 images that already manually perfected!
        if img_num <= 70:
            continue
    except:
        pass # If naming convention is different, just process it

    img_path = os.path.join(RAW_IMAGES_DIR, filename)
    txt_filename = os.path.splitext(filename)[0] + ".txt"
    txt_filepath = os.path.join(YOLO_LABELS_DIR, txt_filename)
    
    # 1. Ask YOLO to find the dials
    results = model(img_path, verbose=False)
    
    yolo_annotations = []
    
    # 2. Extract the boxes and format them for MakeSense.ai
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # YOLO outputs normalized center_x, center_y, width, height natively!
            # We just grab the first item [0] and format it
            cls_id = int(box.cls[0])
            cx, cy, w, h = box.xywhn[0] 
            
            # Format: class_id center_x center_y width height
            yolo_line = f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            yolo_annotations.append(yolo_line)
            
    # 3. Save the new, much smarter .txt file
    with open(txt_filepath, 'w') as f:
        f.write("\n".join(yolo_annotations))
        
    print(f"✅ Processed {filename}: YOLO found {len(yolo_annotations)} dials.")

print("🎉 YOLO Auto-Annotation Complete! Ready for MakeSense.ai review.")