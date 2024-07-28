import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class DepthAnything(torch.nn.Module):
    def __init__(self, size='small'):
        super().__init__()
        self.preprocessor = AutoImageProcessor.from_pretrained(f'LiheYoung/depth-anything-{size}-hf', do_rescale=True, do_resize=False)
        self.core = AutoModelForDepthEstimation.from_pretrained(f'LiheYoung/depth-anything-{size}-hf')
        self._device = torch.device('cpu')

    def forward(self, images):
        # Needs NumPy/PIL images or list of images as input and returns PyTorch tensors.
        z = self.preprocessor(images=images, return_tensors='pt')['pixel_values']
        z = self.core(z.to(self._device)).predicted_depth
        return z.unsqueeze(1)

    def to(self, device, *args, **kwargs):
        self._device = device
        return super().to(device, *args, **kwargs)

