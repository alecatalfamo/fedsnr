import numpy as np

from torch.utils.data import DataLoader

def calculate_snr(image):
    # Calcola la media e la deviazione standard dell'intera immagine
    mean_signal = np.mean(image)
    std_noise = np.std(image)
    
    # Evita la divisione per zero
    if std_noise == 0:
        return float('inf')  # SNR molto alto se non c'è rumore (poco realistico ma utile per evitare errori)
    
    return mean_signal / std_noise

def calculate_cnr(image):
    # In mancanza di due ROI distinte, possiamo approssimare il CNR usando il contrasto tra media e variazione
    mean_signal = np.mean(image)
    std_noise = np.std(image)
    
    # Qui interpretiamo la deviazione standard come una misura del rumore e il contrasto come differenza dalla media
    return mean_signal / std_noise

# Funzione per calcolare i valori medi aggregati di SNR e CNR per un intero dataset
def calculate_dataset_snr_cnr(dataset):
    total_snr = 0
    total_cnr = 0
    num_images = 0
    total_dataset = dataset.dataset  # Se il dataset è un DataLoader, accedi al dataset sottostante
    new_dataloader = DataLoader(total_dataset, batch_size=1)  # Creare un DataLoader per accedere ai singoli elementi
    
    for _, data in enumerate(new_dataloader, 0):  # Assumendo che il dataset sia nel formato (immagine, etichetta)
        image, = data['image']
        image_array = image.numpy()
        
        # Calcolo di SNR e CNR per l'immagine corrente
        snr = calculate_snr(image_array)
        cnr = calculate_cnr(image_array)
        
        total_snr += snr
        total_cnr += cnr
        num_images += 1
    
    # Calcola la media
    mean_snr = total_snr / num_images
    mean_cnr = total_cnr / num_images
    
    return mean_snr, mean_cnr
