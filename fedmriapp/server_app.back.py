import json
import torch.nn.functional as F
import torch
import flwr as fl
import numpy as np
import random
import os
from typing import List, Tuple
from torch.utils.data import DataLoader
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from collections import OrderedDict
from datasets import load_dataset
from pathlib import Path
from fedmriapp.models.models import get_custom_model
from fedmriapp.client_app import apply_transforms
from fedmriapp.mristrategy.fed_mri import FedMRI
from fedmriapp.reproducibility.reproducible_strategy import make_strategy_reproducible

device = "cuda"
TESTSET_PATH = "./fedmriapp/testset/test"
PATH = "./fedmriapp/results"

with open('fedmriapp/fl_config.json') as f:
    fl_config = json.load(f)

result_file = f"{PATH}/{fl_config['strategy']}-C{fl_config['fitFraction']}.csv"

def init_results_file():
    with open(result_file, "w") as f:
        f.write("round,loss,accuracy\n")
    

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

set_all_seeds(42)



def get_testset():
    dataset_path = Path(TESTSET_PATH)
    # Set seeds before dataset loading
    set_all_seeds(42)
    dataset = load_dataset('imagefolder', data_dir=dataset_path)
    test_dataset = dataset['train']
    return test_dataset

def set_weights(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def test(net, testloader, device):
    # Set seeds before testing
    set_all_seeds(42)
    net.eval()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data["image"], data["label"]
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += F.cross_entropy(outputs, labels, reduction="sum").item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= total
    accuracy = correct / total
    return loss, accuracy

def get_evaluate_fn(testloader, device):
    """Return a callback that evaluates the global model."""
    def evaluate(server_round: int, parameters, config):
        # Set seeds before evaluation
        #set_all_seeds(42 + server_round)  # Different seed per round
        set_all_seeds(42)
        model = get_custom_model()
        set_weights(model, parameters)
        loss, accuracy = test(model, testloader, device)
        
        with open(result_file, "a") as f:
            f.write(f"{server_round},{loss},{accuracy}\n")
        
        return loss, {"accuracy": accuracy}

    return evaluate

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    return {"accuracy": sum(accuracies) / total_examples}

def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    b_values = []
    for _, m in metrics:
        my_metric_str = m["my_metric"]
        my_metric = json.loads(my_metric_str)
        b_values.append(my_metric["b"])
    return {"max_b": max(b_values)}

def on_fit_config(server_round: int) -> Metrics:
    lr = 0.01
    if server_round > 2:
        lr = 0.005
    return {"lr": lr}

def server_fn(context: Context):
    """A function that creates the components for a ServerApp."""
    # Set seeds at server initialization
    set_all_seeds(42)
    
    num_rounds = fl_config["numRounds"]
    fraction_fit = fl_config["fitFraction"]
    min_available_clients = fl_config["fitClients"]
    
    ndarrays = get_weights(get_custom_model())
    parameters = ndarrays_to_parameters(ndarrays)

    testset = get_testset()
    # Add seed to DataLoader for reproducible batching
    g = torch.Generator()
    g.manual_seed(42)
    testloader = DataLoader(
        testset.with_transform(apply_transforms), 
        batch_size=32,
        shuffle=False,
        generator=g,
        worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id),
    )

    strategyFedAvg = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=0,
        min_available_clients=min_available_clients,
        initial_parameters=parameters,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=get_evaluate_fn(testloader, device=device),
    )
    
    strategyFedMRI = FedMRI(
        fraction_fit=fraction_fit,
        fraction_evaluate=0,
        min_available_clients=min_available_clients,
        initial_parameters=parameters,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=get_evaluate_fn(testloader, device=device),
    )
    if fl_config["strategy"] == "FedAvg":
        strategy = strategyFedAvg
    elif fl_config["strategy"] == "FedMRI":
        strategy = strategyFedMRI
    else:
        raise ValueError(f"Invalid strategy: {fl_config['strategy']}")
    
    strategy = make_strategy_reproducible(strategy, seed=42)
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp with seed
set_all_seeds(42)
init_results_file()
app = ServerApp(server_fn=server_fn)
