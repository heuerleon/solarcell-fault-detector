from torchvision import datasets, transforms
import torch
# to split test_data always same
torch.manual_seed(1004)

import torch.nn as nn
from torch.utils.data import DataLoader

# you can change input size(don't forget to change linear layer!)
custom_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Standardwerte
])

def make_data_loader(args):
    #calculate_mean_and_std(args)
    # Get Dataset
    dataset = datasets.ImageFolder(args.data, transform=custom_transform)
    
    # split dataset to train/test
    train_data_percentage = 0.8
    train_size = int(train_data_percentage * len(dataset))
    test_size = len(dataset) - train_size
    
    # you must set "seed" to get same test data
    # you can't compare different test set's accuracy
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Get Dataloader
    cores = 48
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=cores, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=cores, shuffle=False, pin_memory=True)
    
    return train_loader, test_loader

def calculate_mean_and_std(args):
    # Load dataset
    dataset = datasets.ImageFolder(args.data, transform=transforms.ToTensor())
    
    # Stack all images into a tensor
    data = torch.cat([img[0].view(3, -1) for img in dataset], dim=1)  # Assuming RGB
    
    # Calculate mean and std per channel
    mean = data.mean(dim=1)
    std = data.std(dim=1)
    
    print("Mean:", mean)
    print("Std Dev:", std)
