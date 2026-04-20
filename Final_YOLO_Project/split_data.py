import os
import random
import shutil

# 1. Setup paths
BASE_DIR = "yolo_dataset"
IMG_DIR = os.path.join(BASE_DIR, "images")
LBL_DIR = os.path.join(BASE_DIR, "labels")

# Create the new Train and Val folders
for folder in ['train', 'val']:
    os.makedirs(os.path.join(BASE_DIR, folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, folder, 'labels'), exist_ok=True)

# 2. Get all images and shuffle them randomly
images = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
random.shuffle(images)

# 3. Calculate the 80/20 split
split_index = int(len(images) * 0.8)
train_imgs = images[:split_index]
val_imgs = images[split_index:]

def move_files(file_list, destination_name):
    for img_name in file_list:
        txt_name = img_name.replace('.jpg', '.txt')
        
        # Move Image
        shutil.move(os.path.join(IMG_DIR, img_name), 
                    os.path.join(BASE_DIR, destination_name, 'images', img_name))
        # Move Text file (if it exists)
        if os.path.exists(os.path.join(LBL_DIR, txt_name)):
            shutil.move(os.path.join(LBL_DIR, txt_name), 
                        os.path.join(BASE_DIR, destination_name, 'labels', txt_name))

# 4. Execute the move
print("🚚 Moving files into Train (80%) and Val (20%) folders...")
move_files(train_imgs, 'train')
move_files(val_imgs, 'val')

print(f"✅ Split Complete! Train: {len(train_imgs)} images | Val: {len(val_imgs)} images")