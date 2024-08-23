import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import timm
import numpy as np
import torch
import models_vit
from RF_loader import RF_loader
device='cuda' if torch.cuda.is_available() else 'cpu'
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

Rf=RF_loader("RETFound_cfp_weights.pth",'org').to(device)
data_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

folder_path='/content/AMD_dataset(CLAHE)'
train_data = datasets.ImageFolder(root=f"{folder_path}/Train", transform=data_transform)
valid_data = datasets.ImageFolder(root=f"{folder_path}/Validation", transform=data_transform)
test_data = datasets.ImageFolder(root=f"{folder_path}/Test", transform=data_transform)

train_dataloader = DataLoader(dataset=train_data, shuffle=False)
val_dataloader = DataLoader(dataset=valid_data, shuffle=False)
test_dataloader = DataLoader(dataset=test_data, shuffle=False)

def extract_features(model=Rf, split=""):
    if split=='train':
        dataloader=train_dataloader
    elif split=='test':
        dataloader=test_dataloader
    elif split=='val':
        dataloader=val_dataloader
    features_list = []
    labels = []
    with torch.inference_mode():
        for X, y in tqdm(dataloader):
            feature = model.forward_features(X.cuda()).squeeze()
            features_list.append(feature.cpu().numpy())
            labels.append(y.cpu().numpy())
    return features_list, labels

# Extract features from datasets
train_features, train_labels = extract_features(Rf, train_dataloader)
valid_features, valid_labels = extract_features(Rf, val_dataloader)
test_features, test_labels = extract_features(Rf, test_dataloader)
