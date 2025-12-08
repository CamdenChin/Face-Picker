"""
Predict with simple Model

"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse


class ImprovedAttractivenessNet(nn.Module):
    def __init__(self):
        super(ImprovedAttractivenessNet, self).__init__()
        self.backbone = models.resnet18(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.backbone(x)


def predict(image_path, model_path='./attractiveness_model_simple.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = ImprovedAttractivenessNet()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Predict
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        score = output.item() * 10
    
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--model', default='./attractiveness_model_simple.pth')
    args = parser.parse_args()
    
    try:
        score = predict(args.image, args.model)
        
        print("="*60)
        print(f"  Attractiveness Score: {score:.2f} / 10")
        print("="*60)
        
        if score >= 8.0:
            print("  🌟 Very attractive!")
        elif score >= 6.5:
            print("  ✨ Attractive!")
        elif score >= 5.0:
            print("  😊 Above average")
        elif score >= 3.5:
            print("  👍 Average")
        else:
            print("  🙂 Below average")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
