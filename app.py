import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Machine Unlearning Inference Engine - DenseNet-201")

# Enable CORS for React Frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust if necessary for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CIFAR-100 Class labels mapping
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# Define Forget Set classes (Classes 0–9 targeted for unlearning)
FORGET_CLASSES = list(range(10))

# Image Preprocessing Pipeline for CIFAR-100 (32x32 tensor scaling & normalization)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

def load_densenet_checkpoint(checkpoint_path):
    """Helper to instantiate DenseNet-201 and load trained unlearning weights."""
    model = models.densenet201(weights=None)
    # Adjust final classifier layer for CIFAR-100 100-class output
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 100)
    
    try:
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f"Successfully loaded checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"Warning: Could not load checkpoint from {checkpoint_path} ({e}). Using initialized weights.")
    
    model.eval()
    return model

print("Loading DenseNet-201 model checkpoints...")
# Update these paths to match your actual checkpoint folder/filenames if they differ
base_model = load_densenet_checkpoint("models/base_densenet201.pth")
unlearned_model = load_densenet_checkpoint("models/unlearned_densenet201.pth")
retrained_model = load_densenet_checkpoint("models/retrained_densenet201.pth")
print("All DenseNet-201 models loaded successfully!")

models_dict = {
    "Base Model (Original)": base_model,
    "Unlearned Model (Targeted)": unlearned_model,
    "Retrained Model (Gold Standard)": retrained_model
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0) # Add batch dimension

        predictions = {}

        with torch.no_grad():
            for name, model in models_dict.items():
                outputs = model(tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                # Get top-3 prediction vectors
                top3_prob, top3_indices = torch.topk(probabilities, 3)
                
                top_3_list = []
                for i in range(3):
                    idx = top3_indices[i].item()
                    top_3_list.append({
                        "class": CIFAR100_CLASSES[idx],
                        "confidence": round(top3_prob[i].item() * 100, 2)
                    })

                pred_idx = top3_indices[0].item()
                pred_class = CIFAR100_CLASSES[pred_idx]
                confidence = round(top3_prob[0].item() * 100, 2)
                is_forget = pred_idx in FORGET_CLASSES

                predictions[name] = {
                    "predicted_class": pred_class,
                    "confidence": confidence,
                    "is_forget_class": is_forget,
                    "top_3": top_3_list
                }

        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)