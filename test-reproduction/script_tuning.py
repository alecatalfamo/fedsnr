import json
import sys
import argparse
import random
def modify_json(input_file, output_file, fitFraction, server_strategy, client_strategy,
                distribution, dataset, percentage_noisy_clients, fit_clients, seed):
    print("Input file:", input_file)
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Set seed
    rng = random.Random(seed)

    # Randomly select a fraction of clients to be noisy

    data['fitFraction'] = fitFraction
    data['serverStrategy'] = server_strategy
    data['clientStrategy'] = client_strategy
    data['distribution'] = distribution
    data['fitClients'] = fit_clients
    data['dataset'] = dataset

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
    argparser.add_argument('--strategy', type=str, required=False, default=None,
                           help='Shorthand: set both serverStrategy and clientStrategy')
    argparser.add_argument('--serverStrategy', type=str, required=False, default=None)
    argparser.add_argument('--clientStrategy', type=str, required=False, default=None)
    argparser.add_argument('--distribution', type=str, required=True)
    argparser.add_argument('--percentage_noisy_clients', type=float, required=True)
    argparser.add_argument('--seed', type=int, required=True)
    argparser.add_argument('--fit_clients', type=int, required=True)
    argparser.add_argument('--dataset', type=str, required=False, default='alzheimer')

    args = argparser.parse_args()

    server_strategy = args.serverStrategy or args.strategy
    client_strategy = args.clientStrategy or args.strategy
    if not server_strategy or not client_strategy:
        argparser.error("Specificare --strategy (vale per entrambi) oppure --serverStrategy e --clientStrategy separatamente.")

    input_file = args.input
    output_file = args.output

    modify_json(input_file, output_file, args.fitFraction, server_strategy, client_strategy,
                args.distribution, args.dataset, args.percentage_noisy_clients,
                args.fit_clients, args.seed)