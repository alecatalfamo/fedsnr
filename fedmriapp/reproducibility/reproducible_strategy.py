import flwr as fl
import random
import json

from flwr.common import FitIns, Parameters, GetPropertiesIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

with open('fedmriapp/fl_config.json') as f:
    fl_config = json.load(f)
    number_partitions = fl_config['fitClients']


# def make_strategy_reproducible(strategy: Strategy, seed: int) -> Strategy:
#     def reproducible_configure_fit(server_round: int, parameters: Parameters, client_manager: ClientManager) -> list[tuple[ClientProxy, FitIns]]:
#         config = strategy.on_fit_config_fn(server_round) if strategy.on_fit_config_fn else {}
#         fit_ins = FitIns(parameters, config)
#         client_manager.wait_for(strategy.min_available_clients)
#         sample_size, _ = strategy.num_fit_clients(client_manager.num_available())
#         random.seed(int(seed + server_round))
#         available_cids = list(client_manager.all())
#         sampled_cids = random.sample(available_cids, sample_size)
#         #print("Available clients:", available_cids)
#         clients = [client_manager.clients[cid] for cid in sampled_cids]
#         # with open("fedmriapp/results/client_ids.txt", "a") as f:
#         #     str_to_write = f"Round {server_round}: {sampled_cids}\n"
#         #     f.write(str_to_write)
#         return [(client, fit_ins) for client in clients]

def make_strategy_reproducible(strategy: Strategy, seed: int) -> Strategy:
        
    def reproducible_configure_fit(server_round: int, parameters: Parameters, client_manager: ClientManager) -> list[tuple[ClientProxy, FitIns]]:
        config = strategy.on_fit_config_fn(server_round) if strategy.on_fit_config_fn else {}
        fit_ins = FitIns(parameters, config)
        get_properties_ins = GetPropertiesIns(fl.common.Config({}))
        client_manager.wait_for(strategy.min_available_clients)
        sample_size, _ = strategy.num_fit_clients(client_manager.num_available())
        random.seed(int(seed + server_round))
        #available_cids = list(client_manager.all())
        # for id_p, client_proxy in enumerate(client_manager.clients.values()):
        #     if len(client_proxy.properties.keys()) == 0:
        #         print("Sto settando la partition_id")
        #         client_proxy.properties['partition_id'] = id_p
        #         print("Client properties", client_proxy.node_id, client_proxy.properties)
        #     #print("Client properties", client_proxy.node_id, client_proxy.properties)
        
        # for client in client_manager.clients.values():    
        #     print("Properties attribute", client.properties)
        #     print("Get Properties result", client.get_properties(get_properties_ins, 6000, 0))
        # #    exit()
        print("Server round", server_round)
        if server_round == 1:
            client_manager.map_cid_client_id = {}
            def new_client_manager_register(self, client: ClientProxy) -> bool:
                print("I'm in client_manager_register")
                if client.cid not in self.map_cid_client_id.values():    
                    client_id = client.get_properties(get_properties_ins, 100, 0).properties['client_id']
                    self.map_cid_client_id[client_id] = client.cid
                    
                if client.cid in self.clients:
                    return False
                
                
                
                self.clients[client.cid] = client
                
                with self._cv:
                    self._cv.notify_all()
            
            def new_client_manager_unregister(self, client: ClientProxy) -> None:
                if client.cid in self.clients:
                    del self.clients[client.cid]
                    for key,values in self.map_cid_client_id.items():
                        if values == client.get_properties(get_properties_ins, 100, 0).properties['client_id']:
                            del self.map_cid_client_id[key]
                            break
                    
                    with self._cv:
                        self._cv.notify_all()
            
            client_manager.register = new_client_manager_register.__get__(client_manager, ClientManager)
            client_manager.unregister = new_client_manager_unregister.__get__(client_manager, ClientManager)
            
            for client in client_manager.clients.values():
                client_manager.register(client)
        
        available_client_ids = list(client_manager.map_cid_client_id.keys())
        available_client_ids = sorted(available_client_ids)
        print("Map CID Client_id", client_manager.map_cid_client_id)
        
        sampled_partitions_id = random.sample(available_client_ids, sample_size)
        
        # with open("fedmriapp/results/client_ids.txt", "a") as f:
        #     str_to_write = f"Round {server_round}: {sampled_partitions_id}\n"
        #     f.write(str_to_write)
        
        clients = [client_manager.clients[client_manager.map_cid_client_id[client_id]] for client_id in sampled_partitions_id]
        
        #print("Properties", clients[0].node_id)
            
        return [(client, fit_ins) for client in clients]

    def reproducible_aggregate_fit(server_round: int, results: list[fl.common.FitRes], failures) -> tuple[Parameters, dict]:
        results.sort(key=lambda x: x[1].metrics["client_id"])
        client_ids = [result[1].metrics['client_id'] for result in results]
        print("Results metrics", client_ids)
        random.seed(int(seed + server_round))
        results.sort(key=lambda x: random.random())
        return original_aggregate_fit(server_round, results, failures)
    
    strategy.configure_fit = reproducible_configure_fit
    original_aggregate_fit = strategy.aggregate_fit
    strategy.aggregate_fit = reproducible_aggregate_fit
    return strategy
