import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def calculate_snr(image):
    """
    Calculate Signal-to-Noise Ratio (SNR) for the entire image.
    
    Args:
    - image: 2D numpy array of the MRI image.
    
    Returns:
    - SNR value.
    """
    mean_signal = np.mean(image)
    noise_region = image[image < mean_signal]  # Background noise
    mean_noise = np.mean(noise_region) if noise_region.size > 0 else 1e-6
    snr = mean_signal / mean_noise
    return snr

def calculate_cnr(image):
    """
    Calculate Contrast-to-Noise Ratio (CNR) for the entire image.
    
    Args:
    - image: 2D numpy array of the MRI image.
    
    Returns:
    - CNR value.
    """
    mean_signal = np.mean(image)
    # Use a predefined threshold to define background; you can adjust this based on your image
    background_threshold = mean_signal * 0.5  # Example threshold for background
    background_region = image[image < background_threshold]

    mean_background = np.mean(background_region) if background_region.size > 0 else 1e-6
    std_background = np.std(background_region) if background_region.size > 0 else 1e-6

    cnr = (mean_signal - mean_background) / std_background if std_background > 0 else 0
    return cnr

# Example usage
if __name__ == "__main__":
    # Load your MRI image (make sure it's grayscale)
    image = io.imread('test.jpg', as_gray=True)

    # Calculate SNR and CNR for the entire image
    snr = calculate_snr(image)
    cnr = calculate_cnr(image)

    print(f"SNR: {snr:.2f}")
    print(f"CNR: {cnr:.2f}")

    # # Optionally display the image
    # plt.imshow(image, cmap='gray')
    # plt.title('MRI Image')
    # plt.axis('off')
    # plt.show()