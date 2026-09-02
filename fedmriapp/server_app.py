import torch
import torch.nn.functional as F
import json
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server.strategy import FedAvg, FedProx, FedAdam
from datasets import load_dataset
from pathlib import Path
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from fedmriapp.models.models import get_custom_model
from fedmriapp.mristrategy.fed_mri import FedSNR
from fedmriapp.reproducibility.reproducible_strategy import make_strategy_reproducible

with open('fedmriapp/fl_config.json') as f:
    fl_config = json.load(f)

PATH = f"./fedmriapp/results/{fl_config['dataset']}"

percentages = len(fl_config['noisyClients']) / fl_config['fitClients']
_server_strat = fl_config['serverStrategy']
_client_strat = fl_config['clientStrategy']
_strat_prefix = f"{_server_strat}-client-{_client_strat}" if _client_strat != _server_strat else _server_strat
result_file = f"{PATH}/{_strat_prefix}-C{fl_config['fitFraction']}-partClients{fl_config['fitClients']}-dist-{fl_config['distribution']}-perc-{percentages}-seed-{fl_config['strategySeed']}.csv"

DATASET = fl_config['dataset']

def init_results_file():
    if not Path(PATH).exists():
        Path(PATH).mkdir(parents=True, exist_ok=True)
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
        model.to(device)
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
    grayscale = Grayscale()
    transforms = ToTensor()

    batch["image"] = [resize(img) for img in batch["image"]]
    batch["image"] = [grayscale(img) for img in batch["image"]]
    batch["image"] = [transforms(img) for img in batch["image"]]
    
    return batch

def get_testset():
    dataset_path = Path("./datasets/global_datasets/alzheimer_dataset") if DATASET == 'alzheimer' else Path("./datasets/global_datasets/brain-tumor-mri")
    dataset = load_dataset('imagefolder', data_dir=dataset_path)
    test_dataset = dataset['test']
    return test_dataset

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
    if fl_config.get("forceCPU", False):
        device = torch.device("cpu")
    else:
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
    strategies = {}

    strategies['FedAvg'] = FedAvg(
        fraction_fit=fl_config["fitFraction"],
        fraction_evaluate=0,
        min_available_clients=fl_config["fitClients"],
        initial_parameters=parameters_param,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=get_evaluate_fn(testloader, device=device),
    )

    strategies['FedSNR'] = FedSNR(
        fraction_fit=fl_config["fitFraction"],
        fraction_evaluate=0,
        min_available_clients=fl_config["fitClients"],
        initial_parameters=parameters_param,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=get_evaluate_fn(testloader, device=device),
    )
    strategies['FedProx'] = FedProx(
        fraction_fit=fl_config["fitFraction"],
        fraction_evaluate=0,
        min_available_clients=fl_config["fitClients"],
        initial_parameters=parameters_param,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=get_evaluate_fn(testloader, device=device),
        proximal_mu=0.01,
    )
    strategies['FedAdam'] = FedAdam(
        fraction_fit=fl_config['fitFraction'],
        fraction_evaluate=0,
        min_available_clients=fl_config['fitClients'],
        initial_parameters=parameters_param,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=get_evaluate_fn(testloader, device=device),
        eta=0.01,
        eta_l=0.01,
        beta_1=0.9,
        beta_2=0.999,
        tau=1e-8,
    )
    
    try:
        strategy = strategies[fl_config["serverStrategy"]]
    except KeyError:
        raise ValueError(f"Invalid server strategy: {fl_config['serverStrategy']}")
    
    strategy = make_strategy_reproducible(strategy, seed=fl_config["strategySeed"])
    
    config = ServerConfig(num_rounds=fl_config["numRounds"])
    return ServerAppComponents(strategy=strategy, config=config)


init_results_file()
app = ServerApp(server_fn=server_fn)
