import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import timm
import numpy as np
import torch
import models_vit
import os 

def RF_loader(model_path, model_type='org',arch='vit_large_patch16'):
    ## model_type : origina weights or fine-tuned
    device='cuda' if torch.cuda.is_available() else 'cpu'
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
      # build model
    model = models_vit.__dict__[arch](
        img_size=224,
        num_classes=2,
        drop_path_rate=0,
        global_pool=False,
    )
    # load model
    if model_type=="org":
        pass
    else : 
        chkpt_dir=os.path.join(model_path,model_type,".pth")
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    return model
