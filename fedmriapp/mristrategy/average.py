import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import filters

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

def threshold_otsu(image):
    """
    Implements Otsu's thresholding method to find an optimal threshold for image segmentation.
    
    Args:
    - image: 2D numpy array of the input image
    
    Returns:
    - Optimal threshold value
    """
    # Compute histogram of pixel intensities
    hist, bin_edges = np.histogram(image.ravel(), bins=256)
    
    # Normalize histogram to get probability distribution
    hist_norm = hist / float(image.size)
    
    # Compute cumulative sum of probabilities
    cumsum = np.cumsum(hist_norm)
    
    # Compute cumulative mean
    cumsum_mean = np.cumsum(hist_norm * np.arange(256))
    
    # Total mean of the image
    total_mean = cumsum_mean[-1]
    
    # Find optimal threshold by maximizing between-class variance
    max_var = 0
    optimal_thresh = 0
    
    for t in range(1, 256):
        # Probabilities of two classes
        w0 = cumsum[t-1]
        w1 = 1 - w0
        
        # Skip if either class is empty
        if w0 == 0 or w1 == 0:
            continue
        
        # Mean of each class
        mu0 = cumsum_mean[t-1] / w0 if w0 > 0 else 0
        mu1 = (total_mean - cumsum_mean[t-1]) / w1 if w1 > 0 else 0
        
        # Between-class variance
        var = w0 * w1 * (mu0 - mu1) ** 2
        
        # Update if a higher variance is found
        if var > max_var:
            max_var = var
            optimal_thresh = t-1
    
    return optimal_thresh

# def calculate_snr(image, border_width=1):
#     """
#     Calculate Signal-to-Noise Ratio (SNR) for an MRI image using background noise from image borders.
    
#     Args:
#     - image: 2D numpy array of the MRI image.
#     - border_width: Width of the border to consider for noise estimation (default: 10 pixels).
    
#     Returns:
#     - SNR value.
#     """
#     #h, w = image.shape
#     h=176 
#     w=208
    
#     #print("image shape", image.shape)
    
#     # Adjust border width to avoid exceeding image dimensions
#     border_width = min(border_width, h//2, w//2)
    
#     # Extract border regions for noise estimation
#     top = image[:border_width, :]
#     bottom = image[-border_width:, :]
#     left = image[border_width:-border_width, :border_width]
#     right = image[border_width:-border_width, -border_width:]
#     noise_region = np.concatenate([top.ravel(), bottom.ravel(), left.ravel(), right.ravel()])
    
#     # Fallback to Otsu's method if border regions are inadequate
#     if len(noise_region) < 50:
#         thresh = threshold_otsu(image)
#         noise_region = image[image <= thresh].ravel()
    
#     # Calculate noise standard deviation
#     std_noise = np.std(noise_region) if noise_region.size > 0 else 1e-6
#     if std_noise <= 1e-6:
#         std_noise = 1e-6
    
#     # Calculate signal mean from central region
#     central_region = image[border_width:-border_width, border_width:-border_width]
#     mean_signal = np.mean(central_region)
    
#     return mean_signal / std_noise

# def calculate_cnr(image):
#     """
#     Calculate Contrast-to-Noise Ratio (CNR) for the entire image.
    
#     Args:
#     - image: 2D numpy array of the MRI image.
    
#     Returns:
#     - CNR value.
#     """
#     mean_signal = np.mean(image)
#     # Use a predefined threshold to define background; you can adjust this based on your image
#     background_threshold = mean_signal * 0.05  # Example threshold for background
#     background_region = image[image < background_threshold]

#     mean_background = np.mean(background_region) if background_region.size > 0 else 1e-6
#     std_background = np.std(background_region) if background_region.size > 0 else 1e-6

#     cnr = (mean_signal - mean_background) / std_background if std_background > 0 else 0
#    return cnr
def calculate_cnr(image, method='otsu', sigma=1.0):
    """
    Compute Contrast-to-Noise Ratio (CNR) for an MRI image using adaptive segmentation.
    
    Parameters:
    -----------
    image : numpy.ndarray
        2D or 3D MRI image
    method : str, optional
        Thresholding method to define regions
        Options: 'otsu', 'mean', 'median'
    sigma : float, optional
        Gaussian filter sigma for noise estimation
    
    Returns:
    --------
    dict containing:
    - CNR value
    - Signal intensities of regions
    - Noise estimation
    """
    # Ensure input is a numpy array
    image = np.array(image, dtype=np.float64)
    
    # Noise estimation using Gaussian filtering
    noise_image = gaussian_filter(image, sigma=sigma)
    noise_std = np.std(noise_image - image)
    
    # Region segmentation based on selected method
    if method == 'otsu':
        # Otsu's thresholding for adaptive segmentation
        threshold = filters.threshold_otsu(image)
        region1 = image[image > threshold]
        region2 = image[image <= threshold]
    elif method == 'mean':
        # Mean-based segmentation
        mean_intensity = np.mean(image)
        region1 = image[image > mean_intensity]
        region2 = image[image <= mean_intensity]
    elif method == 'median':
        # Median-based segmentation
        median_intensity = np.median(image)
        region1 = image[image > median_intensity]
        region2 = image[image <= median_intensity]
    else:
        raise ValueError("Invalid method. Choose 'otsu', 'mean', or 'median'.")
    
    # Compute signal intensities
    signal1_mean = np.mean(region1)
    signal2_mean = np.mean(region2)
    
    # Compute CNR
    cnr = abs(signal1_mean - signal2_mean) / noise_std
    
    return {
        'cnr': cnr,
        'signal1_mean': signal1_mean,
        'signal2_mean': signal2_mean,
        'noise_std': noise_std
    }



# Funzione per calcolare i valori medi aggregati di SNR e CNR per un intero dataset
def calculate_dataset_snr_cnr(dataset, aggregation_method='mean'):
    total_snr = 0
    total_cnr = 0
    num_images = 0
    cnrs = []
    snrs = []
    
    for image in dataset: # Assumendo che il dataset sia nel formato (immagine, etichetta)
        image = image['image']
        image_array = image.numpy()
        
        # Calcolo di SNR e CNR per l'immagine corrente
        snr = calculate_snr(image_array)
        cnr = calculate_cnr(image_array)['cnr']
        
        cnrs.append(cnr)
        snrs.append(snr)

        total_snr += snr
        total_cnr += cnr
        num_images += 1
    
    # Calcola la media
    mean_snr = total_snr / num_images
    mean_cnr = total_cnr / num_images
    
    # Calcola la mediana
    median_snr = np.median(snrs)
    median_cnr = np.median(cnrs)
    
    return (median_snr, median_cnr) if aggregation_method == 'median' else (mean_snr, mean_cnr)