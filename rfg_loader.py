import timm
import torch
def rfg_loader(model_path):

    device='cuda' if torch.cuda.is_available() else 'cpu'
    rfg = timm.create_model('vit_small_patch14_reg4_dinov2',
                            img_size=(392, 392), num_classes=0).eval()
    rfg_weights = torch.load(model_path)
    rfg.load_state_dict(rfg_weights)
    rfg.to(device)
    return rfg