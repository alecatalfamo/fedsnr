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
IFS=',' read -ra distributions <<< "$(read_config 'distributions')"
IFS=',' read -ra percentages <<< "$(read_config 'percentages')"

# Strategia lato server e lato client: se non definite separatamente, usa 'strategies' per entrambe
raw_server=$(read_config 'serverStrategies')
raw_client=$(read_config 'clientStrategies')
raw_both=$(read_config 'strategies')
IFS=',' read -ra serverStrategies <<< "${raw_server:-$raw_both}"
IFS=',' read -ra clientStrategies <<< "${raw_client:-$raw_both}"

cd ..
for fitFraction in "${fitFractions[@]}"; do
    for serverStrategy in "${serverStrategies[@]}"; do
        for clientStrategy in "${clientStrategies[@]}"; do
            for distribution in "${distributions[@]}"; do
                for percentage in "${percentages[@]}"; do

                    python3 test-reproduction/script_tuning.py --input fedmriapp/fl_config.json --output fedmriapp/fl_config.json \
                    --fitFraction $fitFraction --serverStrategy $serverStrategy --clientStrategy $clientStrategy \
                    --distribution $distribution --dataset $dataset \
                    --percentage_noisy_clients $percentage --fit_clients $fit_clients --seed $seed;

                    flwr run;
                done
            done
        done
    done
done
