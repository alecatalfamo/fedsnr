#!/bin/bash
seed=$1
fit_clients=$2
dataset=$3

# Funzione per leggere valori da file INI
read_config() {
    local key=$1
    local config_file="config.ini"
    local values=$(grep "^$key=" "$config_file" | cut -d'=' -f2)
    echo "$values"
}

# Leggi le configurazioni e convertile in array
IFS=',' read -ra fitFractions <<< "$(read_config 'fitFractions')"
IFS=',' read -ra strategies <<< "$(read_config 'strategies')"
IFS=',' read -ra distributions <<< "$(read_config 'distributions')"
IFS=',' read -ra percentages <<< "$(read_config 'percentages')"

cd ..
for fitFraction in "${fitFractions[@]}"; do
    for strategy in "${strategies[@]}"; do
        for distribution in "${distributions[@]}"; do
            for percentage in "${percentages[@]}"; do
                
                python3 test-reproduction/script_tuning.py --input fedmriapp/fl_config.json --output fedmriapp/fl_config.json \
                --fitFraction $fitFraction --strategy $strategy --distribution $distribution --dataset $dataset \
                --percentage_noisy_clients $percentage --fit_clients $fit_clients --seed $seed;

                flwr run;
            done
        done
    done
done
