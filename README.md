# FedSNR
The repo contains the code to reproduce the experiments presented in the paper:


Catalfamo, Alessio, Villari, Massimo, and Galletta, Antonino. <br>
Fedsnr: Enhancing Federated Learning Aggregation Using Signal-to-Noise Ratio in MRI Analysis.
SSRN, 2025. <br>
https://ssrn.com/abstract=5357509 <br>
https://doi.org/10.2139/ssrn.5357509


# Reproduction of the paper tests
If you want to reproduce the tests of the paper, you can follow the steps below.

## Setup Virtual Environment
Create a virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## Generate Partitions of Dataset
Navigate to the folder `test-reproduction`:
```bash
cd test-reproduction
```
Run the partition generator script with 30 partitions (Run both iid and dirichlet to reproduce all the combinations of the paper):
```bash
python3 partition-generator.py --type-partitions-provided [iid|dirichlet] --num-partitions 30 --dataset [alzheimer|brain-tumor]
```

## Partition Server
Run the partition server to serve the generated partitions:
```bash
python3 partition-server.py
```

## Set pyproject.toml for GPU
Open the `fedmriapp/pyproject.toml` file and set the number of GPUs available for client applications.
If you don't have GPUs, set it to 0.
```toml
[tool.flwr.federations.local-simulation-gpu]
options.backend.client-resources.num-cpus = 1 # each ClientApp assumes to use 2CPUs
options.backend.client-resources.num-gpus = <number_of_gpus> # at most 5 ClientApp will run in a given GPU
```

## Toml Supernodes configuration
To reproduce the paper resultsplease open the `fedmriapp/pyproject.toml` file and modify the following line:
```toml
options.num-supernodes= N
```
Where N is 20 for the alzheimer dataset and 30 for the brain-tumor dataset.
Even if during the partition generation you created 30 partitions for both datasets, alzheimer experiments do not exploit the whole dataset.

## Run Experiments
In a new terminal, navigate to the `test-reproduction` folder and run the experiment script:
```bash
./test_exec.sh 42 20 alzheimer
``` 

```bash
./test_exec.sh 42 30 brain-tumor
``` 
for the alzheimer dataset
 