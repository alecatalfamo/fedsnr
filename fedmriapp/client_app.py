import os
import requests
import torch
import numpy as np
import random
import json 
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from datasets import load_from_disk
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from fedmriapp.client.client import FlowerClient
from fedmriapp.models.models import get_custom_model
from fedmriapp.mristrategy.average import calculate_dataset_snr_cnr
from fedmriapp.noise import AddGaussianNoise, AddRicianNoise, AddSaltPepperNoise



def set_all_seeds(seed: int = 42):
    """Set seeds for all random number generators."""
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

# Set global configuration
SERVER_SPLIT = 'localhost'
PORT_FLASK = 5000
FOLDER = Path(os.getcwd() + '/partitions')
# DATASET = 'alzheimer'
DATASET = 'braintumor'
device = "cuda"

with open('fedmriapp/fl_config.json') as fl_config_file:
    fl_config = json.load(fl_config_file)

with open('fedmriapp/client/client_config.json') as client_config_file:
    client_config = json.load(client_config_file)

DISTRIBUTION = fl_config['distribution']
LIST_NOISY_CLIENTS = fl_config['noisyClients']

# Set initial seeds
set_all_seeds(42)

def untar_file(file_path, folder_path):
    """Extract a zip file."""
    import zipfile
    # Set seed before extraction to ensure consistent file handling
    set_all_seeds(42)
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(folder_path)

def request_local_train_data(client_id, folder_path,):
    """Request and save the partition data for a specific client."""
    # Set seed based on client_id for reproducible client-specific behavior
    set_all_seeds(42)
    
    print("Distribution used:", DISTRIBUTION)
    
    if DISTRIBUTION == 'iid':
        url = f'http://{SERVER_SPLIT}:{PORT_FLASK}/partitions/{client_id}?dataset={DATASET}'
    elif DISTRIBUTION == 'dirichlet':
        url = f'http://{SERVER_SPLIT}:{PORT_FLASK}/partitions-dirichlet/{client_id}?dataset={DATASET}'
    
    print("URL:", url)
    
    response = requests.get(url)
    
    client_folder = folder_path / str(client_id)
    client_folder.mkdir(parents=True, exist_ok=True)
    
    zip_file_path = client_folder / f'local_train-{client_id}.zip'
    with open(zip_file_path, 'wb') as file:
        file.write(response.content)
    
    untar_file(str(zip_file_path), str(client_folder))
    full_path = client_folder / f'partition_{client_id}'
    return full_path

class Resize:
    """Resize an image to the specified size."""
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return img.resize(self.size)

class Grayscale:
    """Convert an image to grayscale."""
    def __call__(self, img):
        return img.convert("L")

def apply_transforms(batch, noise_type='rician', noise_params=None):
    """Apply transformations to the image batch including noise."""
    resize = Resize((208, 176)) if DATASET == 'alzheimer' else Resize((244, 244))
    #resize = Resize((244,244))
    grayscale = Grayscale()
    transforms = ToTensor()
    
    # Set default noise parameters if none provided
    if noise_params is None:
        noise_params = {'std': 0.0}  # for Gaussian/Rician
        
    # Initialize noise transform based on type
    if noise_type == 'gaussian':
        noise_transform = AddGaussianNoise(**noise_params)
    elif noise_type == 'rician':
        noise_transform = AddRicianNoise(**noise_params)
    elif noise_type == 'salt_pepper':
        noise_transform = AddSaltPepperNoise(**noise_params)
    else:
        noise_transform = None

    batch["image"] = [resize(img) for img in batch["image"]]
    batch["image"] = [grayscale(img) for img in batch["image"]]
    batch["image"] = [transforms(img) for img in batch["image"]]
    
    if noise_transform:
        batch["image"] = [noise_transform(img) for img in batch["image"]]
    
    return batch
def get_data_loader(client_id, batch_size=32, noise_type='gaussian', noise_params=None):
    """Get DataLoader for the client partition with optional noise."""
    set_all_seeds(42)
    
    full_path = request_local_train_data(client_id, FOLDER)
    dataset = load_from_disk(str(full_path))
    
    # Create transform function with specified noise parameters
    transform_fn = lambda x: apply_transforms(x, noise_type=noise_type, noise_params=noise_params)
    dataset_transformed = dataset.with_transform(transform_fn)
    
    g = torch.Generator()
    g.manual_seed(42)
    
    # dataloader = DataLoader(
    #     dataset_transformed, 
    #     batch_size=batch_size, 
    #     shuffle=True,
    #     generator=g,
    #     worker_init_fn=lambda worker_id: np.random.seed(42 + int(client_id) + worker_id),
    #     num_workers=0
    # )
    
    dataloader = DataLoader(
        dataset_transformed, 
        batch_size=batch_size, 
        shuffle=True,
        generator=g,
        worker_init_fn=lambda worker_id: np.random.seed(42 + int(client_id) + worker_id),
        num_workers=0,
        drop_last=True
    )
    return dataloader

def client_fn(context: Context):
    """Create a Flower client instance."""
    # Set seed based on partition ID for reproducible client initialization
    partition_id = context.node_config["partition-id"]
    if partition_id in LIST_NOISY_CLIENTS:
        noisy_params = {'std': 0.8}
        noise_type = 'rician'
    else:
        noisy_params = None
        noise_type = None
    
    set_all_seeds(42 + partition_id)
    
    # Initialize model with fixed seed
    net = get_custom_model()
    
    # Initialize optimizer with fixed seed
    torch.manual_seed(42)
    
    
    criterion_class = getattr(torch.nn, client_config['criterion'])
    criterion = criterion_class()
    
    # Get the optimizer class dynamically
    optimizer_class = getattr(torch.optim, client_config['optimizer'])
    optimizer = optimizer_class
    
    # Get DataLoader with fixed seed
    trainloader = get_data_loader(partition_id, batch_size=fl_config['batchSize'], noise_type=noise_type, noise_params=noisy_params)
    
    # Calculate SNR/CNR with fixed seed
    set_all_seeds(42)
    mri_parameters = calculate_dataset_snr_cnr(trainloader)
    
    # with open('fedmriapp/results/client_loaded.txt', 'a') as f:
    #     f.write(f"Client {partition_id} loaded\n")
    
    return FlowerClient(partition_id, net, client_config['epochs'], mri_parameters[0], criterion, optimizer, client_config['learning_rate'], trainloader).to_client()

# Set final seed before creating the ClientApp
set_all_seeds(42)
app = ClientApp(client_fn=client_fn)

