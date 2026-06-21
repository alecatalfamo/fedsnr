"""
average_results.py — Aggrega i risultati di run con seed diversi.

Raggruppa i CSV per configurazione (strategia, fitFraction, fitClients,
distribuzione, percentuale di client rumorosi) e calcola media e deviazione
standard di loss e accuracy per ogni round.

Uso:
    python3 average_results.py --results_dir ../fedmriapp/results/<subfolder>
    python3 average_results.py --results_dir ../fedmriapp/results/<subfolder> --output_dir ./averaged
"""

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

# Pattern atteso: {prefix}-C{fraction}-partClients{clients}-dist-{dist}-perc-{perc}-seed-{seed}.csv
FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>.+?)"
    r"-C(?P<fraction>[0-9.]+)"
    r"-partClients(?P<clients>\d+)"
    r"-dist-(?P<dist>[^-]+)"
    r"-perc-(?P<perc>[0-9.]+)"
    r"-seed-(?P<seed>\d+)"
    r"\.csv$"
)


def parse_filename(filename: str):
    """Restituisce (prefix, fraction, clients, dist, perc, seed) oppure None."""
    m = FILENAME_PATTERN.match(filename)
    if not m:
        return None
    return (
        m.group("prefix"),
        m.group("fraction"),
        m.group("clients"),
        m.group("dist"),
        m.group("perc"),
        m.group("seed"),
    )


def load_csv(path: Path) -> list[dict]:
    """Carica un CSV round,loss,accuracy e restituisce una lista di dict."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "round": int(row["round"]),
                "loss": float(row["loss"]),
                "accuracy": float(row["accuracy"]),
            })
    return rows


def average_group(files_rows: list[list[dict]]) -> list[dict]:
    """
    Riceve una lista di run (ognuna è una lista di dict per round).
    Calcola mean e std di loss e accuracy per ogni round comune a tutti i run.
    """
    # Trova i round presenti in tutti i file
    round_sets = [set(r["round"] for r in rows) for rows in files_rows]
    common_rounds = sorted(round_sets[0].intersection(*round_sets[1:]))

    result = []
    for rnd in common_rounds:
        losses = [
            next(r["loss"] for r in rows if r["round"] == rnd)
            for rows in files_rows
        ]
        accuracies = [
            next(r["accuracy"] for r in rows if r["round"] == rnd)
            for rows in files_rows
        ]
        n = len(losses)
        mean_loss = sum(losses) / n
        mean_acc = sum(accuracies) / n

        if n > 1:
            std_loss = (sum((x - mean_loss) ** 2 for x in losses) / (n - 1)) ** 0.5
            std_acc = (sum((x - mean_acc) ** 2 for x in accuracies) / (n - 1)) ** 0.5
        else:
            std_loss = 0.0
            std_acc = 0.0

        result.append({
            "round": rnd,
            "mean_loss": mean_loss,
            "std_loss": std_loss,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "num_seeds": n,
        })
    return result


def write_averaged_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "mean_loss", "std_loss", "mean_accuracy", "std_accuracy", "num_seeds"])
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, summary: list[dict]):
    """Scrive un CSV riassuntivo con tutti i run, inclusa l'ultima accuracy media."""
    fieldnames = ["config", "prefix", "fitFraction", "fitClients", "distribution", "perc_noisy",
                  "num_seeds", "final_mean_accuracy", "final_std_accuracy", "final_mean_loss", "final_std_loss"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def main():
    parser = argparse.ArgumentParser(description="Media dei risultati FL su seed multipli")
    parser.add_argument("--results_dir", required=True, help="Directory con i CSV dei risultati")
    parser.add_argument("--output_dir", default=None,
                        help="Directory di output (default: <results_dir>/averaged)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "averaged"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Raggruppa i file per configurazione (escludendo il seed)
    groups: dict[tuple, list[Path]] = defaultdict(list)
    skipped = []

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".csv"):
            continue
        parsed = parse_filename(fname)
        if parsed is None:
            skipped.append(fname)
            continue
        prefix, fraction, clients, dist, perc, _seed = parsed
        key = (prefix, fraction, clients, dist, perc)
        groups[key].append(results_dir / fname)

    if skipped:
        print(f"File ignorati (formato non riconosciuto): {skipped}")

    summary_rows = []

    for (prefix, fraction, clients, dist, perc), paths in sorted(groups.items()):
        print(f"Configurazione: {prefix} C={fraction} clients={clients} dist={dist} perc={perc} → {len(paths)} seed(s)")

        all_rows = [load_csv(p) for p in paths]
        averaged = average_group(all_rows)

        out_name = f"{prefix}-C{fraction}-partClients{clients}-dist-{dist}-perc-{perc}-AVG.csv"
        write_averaged_csv(output_dir / out_name, averaged)

        if averaged:
            last = averaged[-1]
            config_str = f"{prefix}-C{fraction}-partClients{clients}-dist-{dist}-perc-{perc}"
            summary_rows.append({
                "config": config_str,
                "prefix": prefix,
                "fitFraction": fraction,
                "fitClients": clients,
                "distribution": dist,
                "perc_noisy": perc,
                "num_seeds": last["num_seeds"],
                "final_mean_accuracy": last["mean_accuracy"],
                "final_std_accuracy": last["std_accuracy"],
                "final_mean_loss": last["mean_loss"],
                "final_std_loss": last["std_loss"],
            })

    summary_path = output_dir / "summary.csv"
    write_summary_csv(summary_path, summary_rows)
    print(f"\nCSV medi scritti in: {output_dir}")
    print(f"Riepilogo complessivo: {summary_path}")


if __name__ == "__main__":
    main()
