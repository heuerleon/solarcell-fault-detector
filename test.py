import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn # edited
from tqdm import tqdm
from utils._utils import make_data_loader
from model import BaseModel
from torchvision.models import mobilenet_v3_large

def test(args, data_loader, model):
    true = np.array([])
    pred = np.array([])
    
    model.eval()
    
    pbar = tqdm(data_loader)
    for i, (x, y) in enumerate(pbar):
        
        image = x.to(args.device)
        label = y.to(args.device)                

        output = model(image)
        
        label = label.squeeze()
        output = output.argmax(dim=-1)
        output = output.detach().cpu().numpy()
        pred = np.append(pred,output, axis=0)
        
        label = label.detach().cpu().numpy()
        true =  np.append(true,label, axis=0)
    return pred, true


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2024 DL Term Project')
    parser.add_argument('--model-path', default='checkpoints/model.pth', help="Model's state_dict")
    parser.add_argument('--data', default='data/', type=str, help='data folder')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    num_classes = 2
    
    # hyperparameters
    args.batch_size = 4
    
    # Make Data loader and Model
    _, test_loader = make_data_loader(args)

    # instantiate model
    # model = BaseModel()
    model = mobilenet_v3_large()
    
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 1280),
        nn.Hardswish(),
        nn.Linear(1280, num_classes)  # Binary classification
    )
    
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    
    # Test The Model
    pred, true = test(args, test_loader, model)
        
    accuracy = (true == pred).sum() / len(pred)
    print("Test Accuracy : {:.5f}".format(accuracy))

    