# GCBA & GBA - with latency as  ✨performance metrics✨
from Substrate import SubstrateNetwork, SubstrateNode, VNF, haversine
from env_simulation import generate_dataset
from user_prediction import train_user_prediction_model, predict_user_locations
from KMeans import apply_kmeans_clustering;
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

    successful_embeddings_gcba = 0  # Initialize successful embeddings count

    # 2. Descending-order list of VNF clusters (size ⬇️)
    LV = sorted(all_vnf_clusters, key=lambda cluster: len(cluster), reverse=True)

    while LV:  # check every cluster
        Nu_i = LV.pop(0)  # Get the first cluster in LV
        cluster_latency = 0  # current cluster latency

        for vnf in Nu_i:  # For each VNF in the cluster
            # Embed the VNF to the appropriate substrate node(Ns)
            latency, success = embedding_group(vnf, substrate_network, MP, user_locations)
            successful_embeddings_gcba += 1
            if success:
                successful_embeddings_gcba += 1
            cluster_latency += latency
            MP.add((vnf, vnf.ns))
        # update total latency
        total_latency += cluster_latency
    return MP, total_latency, successful_embeddings_gcba


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
    successful_embeddings_gba = 0  # Initialize successful embeddings count
    # 2. Calculate v(T) for each VNF v in the set of VNFs (neighborhood resource)
    vnf_values = {}  # get v(T) value for all clusters
    for vnf in vnfs:
        vnf_values[vnf] = calculate_neighbor_latency_vt(vnf, substrate_network, user_locations)
    # 3. Descending-order list of VNFs based on v(T) value (V(T) ⬇️)
    sorted_vnfs = sorted(
        vnfs, key=lambda vnf: vnf_values[vnf], reverse=True)
    for vnf in sorted_vnfs:  # Process VNFs one by one in sorted order
        # 4. Embed the VNF to the appropriate substrate node
        latency, success = embedding_group(vnf, substrate_network, MP, user_locations)
        if success:
            successful_embeddings_gba += 1

        MP.add((vnf, vnf.ns))
        total_latency += latency
    return MP, total_latency, successful_embeddings_gba
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
    min_latency = float('inf')
    best_node = None

    for ns in substrate_network.nodes:
        if ns.can_host(vnf):
            latency = np.mean([substrate_network.get_latency(ns, user_loc) for user_loc in user_locations])
            if latency < min_latency:
                min_latency = latency
                best_node = ns

    if best_node is None:
        return float('inf'), False  # failure to embed

    best_node.host(vnf)
    vnf.ns = best_node
    return min_latency, True



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

    # Extract start locations from user_movements
    start_locations = []
    for _, row in user_movements.iterrows():
        point = row['start_point']
        x, y = map(float, point.replace('POINT(', '').replace(')', '').split())
        start_locations.append((x, y))

    # 4️⃣ Initialize substrate network and VNFs
    substrate_network = SubstrateNetwork()
    substrate_network.nodes = base_stations

   #  TODO: Generate and cluster VNFs using K-means
    # vnfs = []
    # vnf_clusters = [[],[]]
    # Generate VNFs
    num_vnfs = 100  # Arbitrary number of VNFs, can be adjusted
    vnfs = []
    for i in range(num_vnfs):
        # Randomly assign CPU requirements (e.g., between 10 and 30)
        cpu_req = np.random.randint(10, 31)
        # Randomly place VNFs within the simulation area (based on base station locations)
        bs_idx = np.random.randint(0, len(base_stations))
        location = base_stations[bs_idx].location  # Use base station location as VNF location
        vnf = VNF(id=i, cpu_req=cpu_req, location=location)
        vnfs.append(vnf)

    # Cluster VNFs using K-means
    # Extract VNF locations for clustering
    vnf_locations = np.array([vnf.location for vnf in vnfs])
    num_clusters = 2  # Example: 2 clusters, can be adjusted
    cluster_labels = apply_kmeans_clustering(vnf_locations, num_clusters)

    # Group VNFs into clusters based on labels
    vnf_clusters = [[] for _ in range(num_clusters)]
    for vnf, label in zip(vnfs, cluster_labels):
        vnf_clusters[label].append(vnf)


    # Run GCBA
    print('👉 Running GCBA...')
    # Before running GCBA
    attempted_embeddings_gcba = 100

    gcba_mp, gcba_latency, successful_embeddings_gcba = group_connectivity_based_algorithm(substrate_network, vnf_clusters, predicted_locations)
    print('👉 GCBA Mapping:')
    print(f'👉 GCBA Total Latency: {gcba_latency:.2f} km')
    for vnf, node in gcba_mp:
        print(f'  VNF ID: {vnf.id}, Mapped to Base Station ID: {node.id}, '
              f'Location: ({node.location[0]:.5f}, {node.location[1]:.5f})')
    print(f'🎉 GCBA Embedding Success Number: {successful_embeddings_gcba}')
    print(f'🎉 GCBA Embedding Success Rate: {successful_embeddings_gcba / attempted_embeddings_gcba * 100:.2f}%')

    # Reset substrate network for GBA
    for node in substrate_network.nodes:
        node.cpu = 100
        node.hosted_vnfs = []

    # Run GBA
    print('👉 Running GBA...')
    attempted_embeddings_gba = 100

    gba_mp, gba_latency, successful_embeddings_gba = group_based_algorithm(substrate_network, vnfs, predicted_locations)
    print(f'👉 GBA Total Latency: {gba_latency:.2f} km')
    print('👉 GBA Mapping:')
    for vnf, node in gba_mp:
        print(f'  VNF ID: {vnf.id}, Mapped to Base Station ID: {node.id}, '
              f'Location: ({node.location[0]:.5f}, {node.location[1]:.5f})')
    print(f'🎉 GCBA Embedding Success Number: {successful_embeddings_gcba}%')
    print(f'🎉 GBA Embedding Success Rate: {successful_embeddings_gba / attempted_embeddings_gba * 100:.2f}%')

if __name__ == '__main__':
    main()
