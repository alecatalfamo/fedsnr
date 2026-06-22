#!/bin/bash
# Deve essere eseguito dalla cartella test-reproduction/
dataset=$1

# Forza thread singolo per riproducibilità cross-machine
# (deve essere settato prima dell'avvio di Python/OMP)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_THREADING_LAYER=sequential

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
IFS=',' read -ra seeds <<< "$(read_config 'seeds')"
IFS=',' read -ra fitClientsList <<< "$(read_config 'fitClientsList')"

# Strategia lato server e lato client: se non definite separatamente, usa 'strategies' per entrambe
raw_server=$(read_config 'serverStrategies')
raw_client=$(read_config 'clientStrategies')
raw_both=$(read_config 'strategies')
IFS=',' read -ra serverStrategies <<< "${raw_server:-$raw_both}"
IFS=',' read -ra clientStrategies <<< "${raw_client:-$raw_both}"

# Avvia il server Flask per le partizioni (usa PWD corrente = test-reproduction/)
python3 server-partition.py &
FLASK_PID=$!
echo "Server partizioni avviato (PID $FLASK_PID)"
sleep 2  # attendi che Flask sia pronto

cleanup() {
    echo "Fermando il server partizioni (PID $FLASK_PID)..."
    kill $FLASK_PID 2>/dev/null
}
trap cleanup EXIT

cd ..
for seed in "${seeds[@]}"; do
    for fit_clients in "${fitClientsList[@]}"; do

        # Aggiorna num-supernodes in pyproject.toml per matchare fit_clients
        sed -i "s/options.num-supernodes = .*/options.num-supernodes = $fit_clients/" pyproject.toml

        for fitFraction in "${fitFractions[@]}"; do
            for serverStrategy in "${serverStrategies[@]}"; do
                for clientStrategy in "${clientStrategies[@]}"; do
                    for distribution in "${distributions[@]}"; do
                        for percentage in "${percentages[@]}"; do

                            python3 test-reproduction/script_tuning.py \
                                --input fedmriapp/fl_config.json \
                                --output fedmriapp/fl_config.json \
                                --fitFraction $fitFraction \
                                --serverStrategy $serverStrategy \
                                --clientStrategy $clientStrategy \
                                --distribution $distribution \
                                --dataset $dataset \
                                --percentage_noisy_clients $percentage \
                                --fit_clients $fit_clients \
                                --seed $seed;

                            flwr run;
                        done
                    done
                done
            done
        done
    done
done
