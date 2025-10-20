import torch
import torch.nn.functional as F
import json
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server.strategy import FedAvg
from datasets import load_dataset
from pathlib import Path
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from fedmriapp.models.models import get_custom_model
from fedmriapp.mristrategy.fed_mri import FedMRI
from fedmriapp.reproducibility.reproducible_strategy import make_strategy_reproducible
#from fedmriapp.client_app import apply_transforms

#TESTSET_PATH = "./fedmriapp/testset/test"
PATH = "./fedmriapp/results"

#DATASET = 'alzheimer'
#DATASET = 'braintumor'

with open('fedmriapp/fl_config.json') as f:
    fl_config = json.load(f)

percentages = len(fl_config['noisyClients']) / fl_config['fitClients']
result_file = f"{PATH}/{fl_config['strategy']}-C{fl_config['fitFraction']}-partClients{fl_config['fitClients']}-dist-{fl_config['distribution']}-perc-{percentages}.csv"

DATASET = fl_config['dataset']

def init_results_file():
    with open(result_file, "w") as f:
        f.write("round,loss,accuracy\n")

def on_fit_config(server_round: int) -> Metrics:
    lr = 0.01 if server_round > 2 else 0.005
    return {"lr": lr}

def test(net, testloader, device):
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
        model = get_custom_model()
        set_weights(model, parameters)
        loss, accuracy = test(model, testloader, device)
        
        with open(result_file, "a") as f:
            f.write(f"{server_round},{loss},{accuracy}\n")
        
        return loss, {"accuracy": accuracy}

    return evaluate

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

def apply_transforms(batch):
    """Apply transformations to the image batch."""    
    resize = Resize((208, 176)) if DATASET == 'alzheimer' else Resize((244, 244))
    #resize = Resize((244, 244))
    grayscale = Grayscale()
    transforms = ToTensor()

    batch["image"] = [resize(img) for img in batch["image"]]
    batch["image"] = [grayscale(img) for img in batch["image"]]
    batch["image"] = [transforms(img) for img in batch["image"]]
    
    return batch

def get_testset():
    dataset_path = Path("./datasets/global_datasets/alzheimer_dataset") if DATASET == 'alzheimer' else Path("./datasets/global_datasets/brain-tumor-mri")
    #dataset_path = Path("/home/fcr/client-mri/brain-tumor-mri")
    dataset = load_dataset('imagefolder', data_dir=dataset_path)
    test_dataset = dataset['test']
    return test_dataset

# def get_weights(model):
#     return [param.cpu().detach().numpy() for param in model.parameters()]

def get_weights(model):
    list_params = []
    for key, value in model.state_dict().items():
        list_params.append(value.cpu().detach().numpy())
    return list_params

def set_weights(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def server_fn(context: Context):
    # Ensure CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_custom_model()
    parameters = get_weights(model)
    parameters_param = ndarrays_to_parameters(parameters)
    
    test_dataset = get_testset()

    testloader = torch.utils.data.DataLoader(
        test_dataset.with_transform(apply_transforms),
        batch_size=32,
        shuffle=False,
    )
    
    strategyFedAvg = FedAvg(
        fraction_fit=fl_config["fitFraction"],
        fraction_evaluate=0,
        min_available_clients=fl_config["fitClients"],
        on_fit_config_fn=on_fit_config,
        initial_parameters=parameters_param,
        evaluate_fn=get_evaluate_fn(testloader, device=device),
    )
    
    strategyMRI = FedMRI(
        fraction_fit=fl_config["fitFraction"],
        fraction_evaluate=0,
        min_available_clients=fl_config["fitClients"],
        on_fit_config_fn=on_fit_config,
        initial_parameters=parameters_param,
        evaluate_fn=get_evaluate_fn(testloader, device=device),
    )
    
    if fl_config["strategy"] == "FedAvg":
        strategy = strategyFedAvg
    elif fl_config["strategy"] == "FedMRI":
        strategy = strategyMRI
    else:
        raise ValueError(f"Invalid strategy: {fl_config['strategy']}")
    
    strategy = make_strategy_reproducible(strategy, seed=fl_config["strategySeed"])
    
    config = ServerConfig(num_rounds=fl_config["numRounds"])
    return ServerAppComponents(strategy=strategy, config=config)


init_results_file()
app = ServerApp(server_fn=server_fn)


