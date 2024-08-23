import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from rfg_loader import rfg_loader
import timm
import torch
device='cuda' if torch.cuda.is_available() else 'cpu'
rfg=rfg_loader("retfoundgreen_statedict.pth")
# Define the data transformation
data_transform = transforms.Compose([
    transforms.Resize(size=(392, 392)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# Load the datasets
folder_path='/content/AMD_dataset(CLAHE)'
train_data = datasets.ImageFolder(root=f"{folder_path}/Train", transform=data_transform)
valid_data = datasets.ImageFolder(root=f"{folder_path}/Validation", transform=data_transform)
test_data = datasets.ImageFolder(root=f"{folder_path}/Test", transform=data_transform)

# Create DataLoaders
train_dataloader = DataLoader(dataset=train_data, shuffle=False)
val_dataloader = DataLoader(dataset=valid_data, shuffle=False)
test_dataloader = DataLoader(dataset=test_data, shuffle=False)

# Define feature extraction function
def extract_features(model, dataloader):
    features_list = []
    labels = []
    with torch.inference_mode():
        for X, y in tqdm(dataloader):
            feature = model.forward_features(X.cuda()).squeeze()
            features_list.append(feature.cpu().numpy())
            labels.append(y.cpu().numpy())
    return features_list, labels

# Extract features from datasets
train_features, train_labels = extract_features(rfg, train_dataloader)
valid_features, valid_labels = extract_features(rfg, val_dataloader)
test_features, test_labels = extract_features(rfg, test_dataloader)
