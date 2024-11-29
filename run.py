import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from model import BaseModel
from tqdm import tqdm
from PIL import Image
import torch.nn as nn 
import sys
torch.manual_seed(0)
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import mobilenet_v3_large

class BinaryImageDataset(Dataset):
    def __init__(self, fault_dir, normal_dir, transform=None):
        self.fault_dir = fault_dir
        self.normal_dir = normal_dir
        self.transform = transform

        self.image_paths = [(os.path.join(fault_dir, f), 0) for f in os.listdir(fault_dir)] + \
                           [(os.path.join(normal_dir, f), 1) for f in os.listdir(normal_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

def compute_accuracy(preds, labels):
    correct = (preds == labels).float().sum()
    return correct / len(preds)

def inference(args, data_loader, model):
    """ model inference """
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        pbar = tqdm(data_loader)
        for images, label in pbar:
            images, label = images.to(args.device), label.to(args.device)
            output = model(images)
            preds.extend(output.argmax(dim=-1).cpu().tolist())
            labels.extend(label.cpu().tolist())
            
    accuracy = compute_accuracy(torch.tensor(preds), torch.tensor(labels))
    return preds, accuracy



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2024 DL Term Project')
    parser.add_argument('--load-model', default='checkpoints/model.pth', help="Model's state_dict")
    parser.add_argument('--batch-size', default=1, help='test loader batch size')
    parser.add_argument('--fault-dir', default='test_image/fault', help='Directory for fault images')
    parser.add_argument('--normal-dir', default='test_image/normal', help='Directory for normal images')

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # ========================== Modify this part  ==========================
    # model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 2)
    model = mobilenet_v3_large()
    
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 1280),
        nn.Hardswish(),
        nn.Linear(1280, 2)  # Binary classification
    )
    # =======================================================================
    
    state_dict = torch.load(args.load_model)  
    model.load_state_dict(state_dict, strict=True)
    
    model.to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Standardwerte
    ])
    
    test_data = BinaryImageDataset(args.fault_dir, args.normal_dir, transform=transform)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    preds, acc = inference(args, test_loader, model)
        
    print(f"Accuracy: {acc * 100:.2f}%")
    with open('result.txt', 'w') as f:
        f.writelines('\n'.join(map(str, preds)))


