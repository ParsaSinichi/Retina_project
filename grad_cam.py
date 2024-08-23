
import torch.nn.functional as F
import math
import argparse
import cv2
import numpy as np
import torch
import os
import cv2
from tqdm import tqdm
from PIL import Image
# import models_vit
import numpy as np
from RF_loader import RF_loader
import timm
import torch
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget, RawScoresOutputTarget, ClassifierOutputSoftmaxTarget


device='cuda' if torch.cuda.is_available() else 'cpu'

def reshape_transform(tensor, height=28, width=28):


    # result = tensor[:, 1:, :].reshape(tensor.size(0),
    #                                   height, width, tensor.size(2))
    result = tensor[:, 5:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    # print(result.shape)
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

methods = \
    {"gradcam": GradCAM,
      "scorecam": ScoreCAM,
      "gradcam++": GradCAMPlusPlus,
      "ablationcam": AblationCAM,
      "xgradcam": XGradCAM,
      "eigencam": EigenCAM,
      "eigengradcam": EigenGradCAM,
      "layercam": LayerCAM,
      "fullgrad": FullGrad}



image_path=""

import torch
device = torch.device('cuda')

model = RF_loader("RETFound_cfp_weights.pth",'org').to(device)
dataset_path = "/content/ADAM_ORG"

target_layers = [model.base_model.blocks[1].norm1]
cam = methods["gradcam"](model=model,
                              target_layers=target_layers,

                              reshape_transform=reshape_transform)


rgb_img = cv2.imread(image_path, 1)[:, :, ::-1] #BGR to RGB
rgb_img = cv2.resize(rgb_img, (224, 224))
org_image = rgb_img
rgb_img = np.float32(rgb_img) / 255
# input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
# std=[0.5, 0.5, 0.5])
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

# If None, returns the map for the highest scoring category.
# Otherwise, targets the requested category.
targets = None
# targets = [ClassifierOutputTarget(1)]
# AblationCAM and ScoreCAM have batched implementations.
# You can override the internal batch size for faster computation.
cam.batch_size = 32
# print(input_tensor.shape)
grayscale_cam = cam(input_tensor=input_tensor,
                    targets=targets,
                    eigen_smooth=True,
                    aug_smooth=True,

                    )

# Here grayscale_cam has only one image in the batch
grayscale_cam = grayscale_cam[0, :]

cam_image = show_cam_on_image(rgb_img, grayscale_cam)
import matplotlib.pyplot as plt
# plt.imshow(cv2.cvtColor(cam_image,cv2.COLOR_BGR2RGB))
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# # Display the images
axs[0].imshow(org_image)
axs[0].axis('off')  # Hide axes
axs[1].imshow(cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB))
axs[1].axis('off')  # Hide axes

# Add a title to the figure
title = 'Non-AMD' if "Non-AMD" in image_path else "AMD"

fig.suptitle(f'{title + "  RETFOUND Green layer1 "}', fontsize=16)

plt.show()
plt.savefig()

# output_image_path = os.path.join(
#     '/content/rfg_targ(1)_head', f"{os.path.splitext(image)[0]}_cam.png")
# plt.savefig(output_image_path)
plt.close(fig)
