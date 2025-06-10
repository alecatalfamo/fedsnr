import flwr as fl
import torch
import os
import random
import numpy as np
import sys
import gc

from collections import OrderedDict
from flwr.common import GetPropertiesIns, GetPropertiesRes, Code

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


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, model, epochs, mean_snr, criterion, optimizer, learning_rate, dataloader):
        self.client_id = client_id
        set_all_seeds(42)
        self.model = model
        self.learning_rate = learning_rate
        self.train_data = dataloader
        #print("Train data", self.train_data)
        self.mean_snr = float(mean_snr)
        self.num_epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = "cuda"

    def get_parameters(self):
        return self.model.get_weights()
    
    def  set_parameters(self, parameters):
        self.set_weights(parameters)
    
    def get_properties(self, config):
        properties = {
            "client_id": self.client_id,
        }
        
        return properties
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        set_all_seeds(42)
        self.train(self.model)
        
        metrics = {
            "client_id": self.client_id,
            "mean_snr": self.mean_snr
        }
        
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()], len(self.train_data), metrics
    
    def set_weights(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def train(self, model):
        set_all_seeds(42)
        # Training loop
        self.optimizer = self.optimizer(model.parameters(), lr=self.learning_rate)
        model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            num_batches = len(self.train_data)

            for i, data in enumerate(self.train_data, 0):
                inputs, labels = data['image'], data['label']

                
                # Move to GPU if available
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Accumulate loss
                running_loss += loss.item()

                # Overwriting print statement
                progress = (i + 1) / num_batches * 100  # Progress percentage
                sys.stdout.write(f"\rEpoch [{epoch+1}/{self.num_epochs}] | Batch [{i+1}/{num_batches}] | Loss: {running_loss / (i + 1):.4f} | Progress: {progress:.1f}%")
                sys.stdout.flush()
                
                del inputs, labels, outputs
                torch.cuda.empty_cache()
                gc.collect()

            # Print final loss for the epoch (newline to prevent overwriting)
            print(f"\nEpoch [{epoch+1}/{self.num_epochs}] completed - Avg Loss: {running_loss / num_batches:.4f}")

        print("Finished Training 🎉")
        return 