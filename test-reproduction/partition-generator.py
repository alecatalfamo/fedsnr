from datasets import load_dataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
import zipfile
import os
import argparse
import numpy as np
import random

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--type-partitions-provided", type=str, default="iid")
parser.add_argument("--alpha", type=float, default=0.5, required=False)
parser.add_argument("--num-partitions", type=int, default=40)
parser.add_argument("--noise", action="store_true")
parser.add_argument("--noise-level", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--dataset", type=str, default="alzheimer", help="Path to the dataset directory")
args = parser.parse_args()

# Set random seeds for reproducibility
def set_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    
    # If you're using PyTorch or TensorFlow, you might want to set their seeds as well
    # import torch
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

# Set seeds at the beginning of the script
set_seeds(args.seed)

# Constants

NUM_PARTITIONS = args.num_partitions

folder = "../datasets/global_datasets/alzheimer_dataset" if args.dataset == "alzheimer" else "../datasets/global_datasets/brain-tumor-mri"

# Load dataset from directory
dataset_dict = load_dataset("imagefolder", data_dir=folder)
trainset = dataset_dict["train"]
print("Train Dataset:", trainset)

# Set Alpha parameter
if args.type_partitions_provided == "dirichlet":
    if args.alpha < 0:
        raise ValueError("Alpha parameter must be greater than 0")
    elif args.alpha is None:
        alpha_parameter = 1.5
    else:
        alpha_parameter = args.alpha
else:
    alpha_parameter = 10000000

# Initialize partitioner
partitioner = DirichletPartitioner(
    num_partitions=NUM_PARTITIONS, 
    partition_by="label",
    alpha=alpha_parameter, 
    min_partition_size=10,
    self_balancing=True, 
    shuffle=True
)
partitioner.dataset = trainset

# Determine directory for partitions
if args.type_partitions_provided == "dirichlet":
    directory_partitions = f"./partitions-{args.dataset}-dirichlet"
elif args.type_partitions_provided == "iid":
    directory_partitions = f"./partitions-{args.dataset}-iid"
else:
    raise ValueError("Invalid type of partitions provided")

# Add noise flag to directory if needed
if args.noise:
    directory_partitions = f"{directory_partitions}-noisy"

# Ensure the partition directory exists
os.makedirs(directory_partitions, exist_ok=True)

# Create and save partitions
for i in range(NUM_PARTITIONS):
    partition = partitioner.load_partition(i)
    print(f"Partition {i} size: {len(partition)}")
    
    file_partition = f"{directory_partitions}/partition_{i}"
    
    partition.save_to_disk(file_partition)
    print(f"Partition {i} saved")
    
    # Folder to be zipped
    parent_folder = os.path.dirname(file_partition)
    folder_to_zip = file_partition
    zip_file_name = file_partition + ".zip"
    
    # Create a zip file and add the folder contents
    with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_to_zip):
            for file in files:
                file_path = os.path.join(root, file)
                # Add file to the zip with a relative path
                arcname = os.path.relpath(file_path, start=parent_folder)
                zipf.write(file_path, arcname)
    
    print(f"{folder_to_zip} has been zipped as {zip_file_name}")