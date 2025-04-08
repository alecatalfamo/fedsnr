import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F

class AddGaussianNoise:
    """Add Gaussian noise to an image."""
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std
        
    def __call__(self, img):
        """
        Args:
            img: PIL Image or Tensor
        Returns:
            PIL Image or Tensor: Image with added Gaussian noise
        """
        if isinstance(img, Image.Image):
            img = F.to_tensor(img)
            
        noise = torch.randn_like(img) * self.std + self.mean
        noisy_img = img + noise
        noisy_img = torch.clamp(noisy_img, 0, 1)  # Ensure values stay in valid range
        
        if isinstance(img, Image.Image):
            return F.to_pil_image(noisy_img)
        return noisy_img

class AddRicianNoise:
    """Add Rician noise to an image (specific to MRI)."""
    def __init__(self, std=0.1):
        self.std = std
        
    def __call__(self, img):
        """
        Args:
            img: PIL Image or Tensor
        Returns:
            PIL Image or Tensor: Image with added Rician noise
        """
        if isinstance(img, Image.Image):
            img = F.to_tensor(img)
            
        # Rician noise is created by adding Gaussian noise to real and imaginary parts
        noise_real = torch.randn_like(img) * self.std
        noise_imag = torch.randn_like(img) * self.std
        
        # Calculate magnitude of complex noise
        noisy_img = torch.sqrt((img + noise_real)**2 + noise_imag**2)
        noisy_img = torch.clamp(noisy_img, 0, 1)
        
        if isinstance(img, Image.Image):
            return F.to_pil_image(noisy_img)
        return noisy_img

class AddSaltPepperNoise:
    """Add Salt & Pepper noise to an image."""
    def __init__(self, prob=0.05):
        self.prob = prob
        
    def __call__(self, img):
        """
        Args:
            img: PIL Image or Tensor
        Returns:
            PIL Image or Tensor: Image with added Salt & Pepper noise
        """
        if isinstance(img, Image.Image):
            img = F.to_tensor(img)
            
        noise_mask = torch.rand_like(img)
        salt = (noise_mask < self.prob/2).float()
        pepper = (noise_mask > (1 - self.prob/2)).float()
        
        noisy_img = img * (~(salt.bool() | pepper.bool())).float() + salt
        
        if isinstance(img, Image.Image):
            return F.to_pil_image(noisy_img)
        return noisy_img