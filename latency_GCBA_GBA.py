# GCBA & GBA - with latency as  ✨performance metrics✨
from Substrate import SubstrateNetwork, SubstrateNode, VNF, haversine
from env_simulation import generate_dataset
from user_prediction import train_user_prediction_model, predict_user_locations
import pandas as pd;
import numpy as np

"""
Adapted from the paper:
x(T) = Latency (when resouce is enough)
V(T) = resource / average latency
"""

####################### ALGORITHM #######################
# GCBA 
"""
1. cluster the VNFs into groups
2. sort the clusters in descending order of size
3. choose node for each VNF in the cluster
"""
def group_connectivity_based_algorithm(substrate_network, all_vnf_clusters, user_locations):
    MP = set()  # 1. Initialize an empty set for mappings
    total_latency = 0  # Initialize total latency

    # 2. Descending-order list of VNF clusters (size ⬇️)
    LV = sorted(all_vnf_clusters, key=lambda cluster: len(cluster), reverse=True)

    while LV:  # check every cluster
        Nu_i = LV.pop(0)  # Get the first cluster in LV
        cluster_latency = 0  # current cluster latency

        for vnf in Nu_i:  # For each VNF in the cluster
            # Embed the VNF to the appropriate substrate node(Ns)
            latency = embedding_group(vnf, substrate_network, MP, user_locations)
            cluster_latency += latency
            MP.add((vnf, vnf.ns))
        # update total latency
        total_latency += cluster_latency
    return MP, total_latency


# GBA
"""
1. No Cluster in GBA!!!! only focus on individual VNFs
2. Calculate v(T)[neighborhood resource] for each VNF v in the set of VNFs 
   ➡️ (v(T) = resource / average latency)
3. Process VNFs with higher v(T) first
"""
def group_based_algorithm(substrate_network, vnfs, user_locations):
    MP = set()  # 1. Initialize an empty set for mappings
    total_latency = 0
    # 2. Calculate v(T) for each VNF v in the set of VNFs (neighborhood resource)
    vnf_values = {}  # get v(T) value for all clusters
    for vnf in vnfs:
        vnf_values[vnf] = calculate_neighbor_latency_vt(vnf, substrate_network, user_locations)
    # 3. Descending-order list of VNFs based on v(T) value (V(T) ⬇️)
    sorted_vnfs = sorted(
        vnfs, key=lambda vnf: vnf_values[vnf], reverse=True)
    for vnf in sorted_vnfs:  # Process VNFs one by one in sorted order
        # 4. Embed the VNF to the appropriate substrate node
        latency = embedding_group(vnf, MP, substrate_network)
        MP.add((vnf, vnf.ns))
        total_latency += latency
    return MP, total_latency
##################################################################


# Calculate neighborhood latency value  (v(T))
def calculate_neighbor_latency_vt(vnf, substrate_network, MP, user_locations):
    total_resources = 0
    total_latency = 0
    neighbor_count = 0
    # get latency of all neighbors
    for node in substrate_network.nodes:
        if node.can_host(vnf):  # does the node have enough resources?
            resources = node.cpu # resource
            latency = np.mean([substrate_network.get_latency(node, user_loc) for user_loc in user_locations])
            total_resources += resources
            total_latency += latency
            neighbor_count += 1

    if neighbor_count == 0:
        return 0
    # v(T) = resource / average latency
    return total_resources / (total_latency / neighbor_count)


# Choose position for VNF (by x(T)) => Lantency
def embedding_group(vnf, substrate_network, MP, user_locations):
    # min_latency was a very large number, make it smaller in the process
    min_latency = float('inf')
    best_node = None

    for ns in substrate_network.nodes:
        if ns.can_host(vnf):  # does the node have enough resources?
            # calculate latency to user locations (mean) and find the best ns
            latency = np.mean([substrate_network.get_latency(ns, user_loc) for user_loc in user_locations])
            if latency < min_latency:
                min_latency = latency
                best_node = ns

    if best_node is None:
        raise Exception("No suitable substrate node found for VNF")

    # VNF => best substrate node
    best_node.host(vnf)
    vnf.ns = best_node  # best substrate node
    return min_latency  # minimum latency


def main():
    # 1️⃣ Simulate environment
    print('👉 Starting dataset generation...')
    user_movements, base_stations = generate_dataset(610000, 50)
    user_movements.to_csv('./datasets/simulated_sf_users.csv', index=False)
    base_station_df = pd.DataFrame([(bs.location[0], bs.location[1]) for bs in base_stations], columns=['x', 'y'])
    base_station_df.to_csv('./datasets/simulated_base_stations.csv', index=False)
    print('👉 Environment simulation completed!')

    # 2️⃣ Train user prediction model
    model = train_user_prediction_model('./datasets/simulated_sf_users.csv')

    # 3️⃣ Predict user end locations
    predicted_locations = predict_user_locations(model, user_movements)

    # 4️⃣ Initialize substrate network and VNFs
    substrate_network = SubstrateNetwork()
    substrate_network.nodes = base_stations
    # TODO: Generate VNF clusters here
    """
    # generate VNF clusters
    # ...
    # ...
    # ...
    # vnf_clusters = [[], [], [], ...]
    """
    vnf_clusters = [[], [], [], ...]
 

    # Run GCBA
    print('👉 Running GCBA...')
    gcba_mp, gcba_latency = group_connectivity_based_algorithm(substrate_network, vnf_clusters, predicted_locations)
    print('👉 GCBA Mapping:')
    print(f'👉 GCBA Total Latency: {gcba_latency:.2f} km')
    for vnf, node in gcba_mp:
        print(f'  VNF ID: {vnf.id}, Mapped to Base Station ID: {node.id}, '
              f'Location: ({node.location[0]:.5f}, {node.location[1]:.5f})')

    # Reset substrate network for GBA
    for node in substrate_network.nodes:
        node.cpu = 100
        node.hosted_vnfs = []

    # Run GBA
    print('👉 Running GBA...')
    gba_mp, gba_latency = group_based_algorithm(substrate_network, sum(vnf_clusters, []), predicted_locations)
    print(f'👉 GBA Total Latency: {gba_latency:.2f} km')
    print('👉 GBA Mapping:')
    for vnf, node in gba_mp:
        print(f'  VNF ID: {vnf.id}, Mapped to Base Station ID: {node.id}, '
              f'Location: ({node.location[0]:.5f}, {node.location[1]:.5f})')

if __name__ == '__main__':
    main()