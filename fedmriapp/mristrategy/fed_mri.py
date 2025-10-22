from typing import List, Tuple, Dict, Any
import flwr as fl
from flwr.common import FitRes

from numpy import ndarray as NDArrays
from flwr.server.client_proxy import ClientProxy
import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import logging

logging.basicConfig(level=logging.INFO)

class FedSNR(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[object, Dict[str, Any]]:
        
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}
        
        # Custom aggregation logic
        aggregated_ndarrays = self.aggregate_inplace(results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        logging.info(f"Aggregated parameters at round {rnd} using FedMRI strategy.")
        return parameters_aggregated, {}
    
    def aggregate_inplace(self, results: list[tuple[ClientProxy, FitRes]]) -> NDArrays:
        # Extract SNR values (ensure metric name matches what clients report)
        snr_values = [fit_res.metrics["mean_snr"] for _, fit_res in results]
        total_snr = sum(snr_values)
        
        # Compute scaling factors based purely on SNR
        scale_factors = [snr / total_snr for snr in snr_values]
        
        # Convert parameters to NDArrays
        parameters_array = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        
        # Weighted average using SNR scaling factors
        aggregated_ndarrays = [
            np.zeros_like(layer) for layer in parameters_array[0]
        ]
        for client_weights, scale in zip(parameters_array, scale_factors):
            for i, layer in enumerate(client_weights):
                aggregated_ndarrays[i] += layer * scale
                
        return aggregated_ndarrays