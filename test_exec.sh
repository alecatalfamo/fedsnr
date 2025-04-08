#!/bin/bash
seed=$1
fit_clients=$2
#percentages=0
# fitFraction=0.7
# strategy=FedMRI
# distribution=dirichlet
# percentages=0.5

for fitFraction in 0.1 0.3 0.5 0.7; do
    for strategy in FedAvg FedMRI; do
        for distribution in iid dirichlet; do
            for percentages in 0.0 0.2 0.5; do
                
                    python3 script_tuning.py --input fedmriapp/fl_config.json --output fedmriapp/fl_config.json \
                    --fitFraction $fitFraction --strategy $strategy --distribution $distribution \
                    --percentage_noisy_clients $percentages --fit_clients $fit_clients --seed $seed;

                    flwr run;        
            done
        done
    done
done

# fitFraction=0.1
# # strategy=FedAvg
# distribution=dirichlet
# # percentages=0.0

# for strategy in FedAvg FedMRI; do
#     for percentages in 0.0 0.1 0.2 0.5; do
#         python3 script_tuning.py --input fedmriapp/fl_config.json --output fedmriapp/fl_config.json \
#                     --fitFraction $fitFraction --strategy $strategy --distribution $distribution \
#                     --percentage_noisy_clients $percentages --fit_clients $fit_clients --seed $seed;
#                     flwr run; 
#     done
# done