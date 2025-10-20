# FedSNR
The repo contains the code to reproduce the experiments presented in the paper:

Catalfamo, Alessio and Villari, Massimo and Galletta, Antonino, Fedsnr: Enhancing Federated Learning Aggregation Using Signal-to-Noise Ratio in MRI Analysis. Available at SSRN: https://ssrn.com/abstract=5357509 or http://dx.doi.org/10.2139/ssrn.5357509

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

Run the partition generator script with the desired parameters:
```bash
python3 partition-generator.py --type-partitions-provided [iid|dirichlet] --num-partitions 30 --dataset [alzheimer|brain-tumor]
```

## Partition Server
Run the partition server to serve the generated partitions:
```bash
python3 partition-server.py
```

## Run Experiments
In a new terminal, navigate to the `test-reproduction` folder and run the experiment script:
```bash
./test_exec.sh
``` 

 