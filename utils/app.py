import torch
import cv2
import numpy as np
from .model import DepthAnything

def get_depth_pipeline(path, device):
    model = DepthAnything('small').to(device)
    model.eval()
    if isinstance(path, str):
        try:
            model.load_state_dict(torch.load(path, device))
        except OSError:
            print('Unable to load', path)

    @torch.no_grad()
    def predict(images):
        pred = model(images)
        pred = pred.squeeze().cpu().numpy()
        return pred

    return predict

def upscale(image, size):
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)

def downscale(image, size): 
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)

def center_crop(image):
    old_x, old_y, _ = image.shape
    crop_size = min(old_x, old_y)
    from_x = old_x // 2 - (crop_size // 2)
    from_y = old_y // 2 - (crop_size // 2)
    to_x = from_x + crop_size
    to_y = from_y + crop_size
    return image[from_x:to_x, from_y:to_y, :]
    
def normalize_depth(depth):
    depth_min = depth.min()
    depth_max = depth.max()
    depth = 255 * (depth - depth_min) / (depth_max - depth_min)
    depth = depth.clip(0, 255)
    depth = depth.astype(np.uint8)
    return depth

def depth2xyz(depth, scale_factor=10):
    h, w = depth.shape
    x, y = np.meshgrid(np.arange(h), np.arange(w))
    z = depth * scale_factor
    return np.stack((x, y, z), axis=-1)

