import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys
from pathlib import Path


# 0. Define the PATH of the .pth file absolutely

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "water_meter_brain.pth"

# 1. Setup
#MODEL_PATH = 'water_meter_brain.pth'
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# 2. Load the Brain
print("🧠 Loading the AI brain...")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Rebuild the ResNet18 structure
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10) 

# Load your custom trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model = model.to(device)
model.eval() # Set to evaluation mode (no learning, just guessing)

# 3. Format the incoming image (Must match validation rules exactly)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_digit(img):
    try:
        # Open and format the image
        #img = Image.open(image_path).convert('RGB')
        img_img = Image.fromarray(img).convert('RGB')
        img_tensor = preprocess(img_img).unsqueeze(0).to(device)
        
        # Ask the AI
        with torch.no_grad():
            outputs = model(img_tensor)
            
            # Calculate confidence percentages
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            
            guess = CLASSES[predicted_idx.item()]
            conf_percent = confidence.item() * 100
            
            #print(f"\n📸 Image: {image_path}")
            print(f"🎯 AI Prediction: It is a '{guess}'")
            print(f"📊 Confidence: {conf_percent:.2f}% sure\n")
            return guess, conf_percent
            
    except Exception as e:
        print(f"❌ Error processing image: {e}")

# Run the test if an image path is provided
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py <path_to_image>")
    else:
        predict_digit(sys.argv[1])