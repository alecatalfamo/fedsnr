#!/bin/bash
# Deve essere eseguito dalla cartella test-reproduction/
dataset=$1

# Forza thread singolo per riproducibilità cross-machine
# (deve essere settato prima dell'avvio di Python/OMP)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_THREADING_LAYER=sequential
export MKL_CBWR=COMPATIBLE
export CUBLAS_WORKSPACE_CONFIG=:4096:8

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

# Costruisce le coppie (serverStrategy, clientStrategy):
#   - se serverStrategies E clientStrategies sono entrambe settate → prodotto cartesiano
#   - altrimenti usa 'strategies' in modalità diagonale (zip): FedSNR+FedSNR, FedAvg+FedAvg, ecc.
raw_server=$(read_config 'serverStrategies')
raw_client=$(read_config 'clientStrategies')
raw_both=$(read_config 'strategies')

declare -a stratPairs=()
if [[ -n "$raw_server" && -n "$raw_client" ]]; then
    IFS=',' read -ra srvs <<< "$raw_server"
    IFS=',' read -ra clts <<< "$raw_client"
    for s in "${srvs[@]}"; do
        for c in "${clts[@]}"; do
            stratPairs+=("${s// /}:${c// /}")
        done
    done
else
    IFS=',' read -ra strats <<< "$raw_both"
    for s in "${strats[@]}"; do
        stratPairs+=("${s// /}:${s// /}")
    done
fi

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

# Leggi numRounds da fl_config.json una volta sola
num_rounds=$(python3 -c "import json; print(json.load(open('fedmriapp/fl_config.json'))['numRounds'])")

for seed in "${seeds[@]}"; do
    for fit_clients in "${fitClientsList[@]}"; do

        # Aggiorna num-supernodes in pyproject.toml per matchare fit_clients
        sed -i "s/options.num-supernodes = .*/options.num-supernodes = $fit_clients/" pyproject.toml

        for fitFraction in "${fitFractions[@]}"; do
            for pair in "${stratPairs[@]}"; do
                serverStrategy="${pair%%:*}"
                clientStrategy="${pair##*:}"
                for distribution in "${distributions[@]}"; do
                    for percentage in "${percentages[@]}"; do

                        # Salta se il CSV esiste già con tutti i round completati
                        if [[ "$serverStrategy" != "$clientStrategy" ]]; then
                            strat_prefix="${serverStrategy}-client-${clientStrategy}"
                        else
                            strat_prefix="$serverStrategy"
                        fi
                        expected_file="fedmriapp/results/$dataset/${strat_prefix}-C${fitFraction}-partClients${fit_clients}-dist-${distribution}-perc-${percentage}-seed-${seed}.csv"
                        if [[ -f "$expected_file" ]] && [[ $(wc -l < "$expected_file") -ge $((num_rounds + 1)) ]]; then
                            echo "Skip: $expected_file"
                            continue
                        fi

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

echo "Calcolo medie dei risultati..."
python3 test-reproduction/average_results.py --results_dir fedmriapp/results/$dataset
echo "Done."
