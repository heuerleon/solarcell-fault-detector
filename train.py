import argparse

import numpy as np
from tqdm import tqdm
from utils._utils import make_data_loader
from model import BaseModel

import torch
import torch.nn as nn # edited
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# wsl
#import torch.multiprocessing as mp
#mp.set_start_method('fork', force=True)

def acc(pred,label):
    pred = pred.argmax(dim=-1)
    return torch.sum(pred == label).item()

def train(args, data_loader, model):
    """
    TODO: Change the training code as you need. (e.g. different optimizer, different loss function, etc.)
            You can add validation code. -> This will increase the accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    for epoch in range(args.epochs):
        train_losses = [] 
        train_acc = 0.0
        total=0
        print(f"[Epoch {epoch+1} / {args.epochs}]")
        
        model.train()
        pbar = tqdm(data_loader)
        for i, (x, y) in enumerate(pbar):
            image = x.to(args.device)
            label = y.to(args.device)
            optimizer.zero_grad()

            output = model(image)
            
            label = label.squeeze()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            total += label.size(0)

            train_acc += acc(output, label)

        scheduler.step()

        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_acc/total
        
        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_loss}')
        print('train_accuracy : {:.3f}'.format(epoch_train_acc*100))
        
        torch.save(model.state_dict(), f'{args.save_path}/model.pth')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='2024 DL Term Project')
    parser.add_argument('--save-path', default='checkpoints/', help="Model's state_dict")
    parser.add_argument('--data', default='data/', type=str, help='data folder')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    num_classes = 2 # edited
    
    """
    TODO: You can change the hyperparameters as you wish.
            (e.g. change epochs etc.)
    """
    
    # hyperparameters
    args.epochs = 5
    args.learning_rate = 1e-3
    args.batch_size = 128
    args.weight_decay = 1e-5

    # check settings
    print("==============================")
    print("Save path:", args.save_path)
    print('Using Device:', device)
    print('Number of usable GPUs:', torch.cuda.device_count())
    
    # Print Hyperparameter
    print("Batch_size:", args.batch_size)
    print("learning_rate:", args.learning_rate)
    print("Epochs:", args.epochs)
    print("==============================")
    
    # Make Data loader and Model
    train_loader, _ = make_data_loader(args)

    # custom model
    # model = BaseModel()
    
    # torchvision model
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights)
    
    # you have to change num_classes to 2
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    #model.classifier.add_module('Dropout', nn.Dropout(p=0.3))
    model.to(device)
    print(model)

    # Training The Model
    train(args, train_loader, model)