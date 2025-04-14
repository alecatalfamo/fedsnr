import json
import sys
import argparse
import random
def modify_json(input_file, output_file, fitFraction, strategy, distribution, percentage_noisy_clients, fit_clients, seed):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Set seed
    rng = random.Random(seed)  
    
    # Randomly select a fraction of clients to be noisy
    
    data['fitFraction'] = fitFraction
    data['strategy'] = strategy
    data['distribution'] = distribution
    data['fitClients'] = fit_clients
    
    list_noise_clients = rng.sample(range(0, data['fitClients']), int(data['fitClients'] * percentage_noisy_clients))
    data['noisyClients'] = list_noise_clients
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: python script_tuning.py <input_json> <output_json>")
    #     sys.exit(1)
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input', type=str, required=True)
    argparser.add_argument('--output', type=str, required=True)
    argparser.add_argument('--fitFraction', type=float, required=True)
    argparser.add_argument('--strategy', type=str, required=True)
    argparser.add_argument('--distribution', type=str, required=True)
    argparser.add_argument('--percentage_noisy_clients', type=float, required=True)
    argparser.add_argument('--seed', type=int, required=True)
    argparser.add_argument('--fit_clients', type=int, required=True)
    
    args = argparser.parse_args()
    input_file = args.input
    output_file = args.output
    
    modify_json(input_file, output_file, args.fitFraction, args.strategy, args.distribution, args.percentage_noisy_clients, args.fit_clients, args.seed)