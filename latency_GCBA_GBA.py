# GCBA & GBA - with latency as  ✨performance metrics✨

####################### ALGORITHM #######################
# GCBA with latency optimization
def group_connectivity_based_algorithm(substrate_network, vnf_set):
    MP = set()  # 1. Initialize an empty set for mappings
    total_latency = 0  # Initialize total latency

    # 2. Descending-order list of VNF clusters (size ⬇️)
    LV = sorted(vnf_set, key=lambda cluster: len(cluster), reverse=True)

    while LV:  # check every cluster
        Nu_i = LV.pop(0)  # Get the first cluster in LV
        cluster_latency = 0  # current cluster latency

        for vnf in Nu_i:  # For each VNF in the cluster
            # Embed the VNF to the appropriate substrate node(Ns)
            latency = embedding_group_with_latency(vnf, substrate_network, MP)
            cluster_latency += latency
            MP.add((vnf, vnf.ns))
        # update total latency
        total_latency += cluster_latency
    return MP, total_latency


# GBA
def group_based_algorithm(substrate_network, vnf_set):
    MP = set()  # 1. Initialize an empty set for mappings
    total_latency = 0

    # 2. Calculate v(T) for each VNF v in the set of VNFs (neighborhood resource)
    vnf_values = {}  # v(T)
    for vnf in vnf_set:
        vnf_values[vnf] = calculate_neighbor_latency_vt(vnf)

    # 3. Descending-order list of VNFs based on v(T) value (V(T) ⬇️)
    sorted_vnfs = sorted(
        vnf_set, key=lambda vnf: vnf_values[vnf], reverse=True)
    for vnf in sorted_vnfs:  # Process VNFs one by one in sorted order
        # 4. Embed the VNF to the appropriate substrate node
        latency = embedding_group(vnf, MP, substrate_network)
        MP.add((vnf, vnf.ns))
        total_latency += latency
    return MP, total_latency
##################################################################


# Calculate neighborhood latency value  (v(T))
def calculate_neighbor_latency_vt(vnf, substrate_network, user_location):
    total_resources = 0
    total_latency = 0
    neighbor_count = 0

    for node in substrate_network.nodes:
        if node.can_host(vnf):  # does the node have enough resources?
            resources = node.cpu # resource
            latency = substrate_network.get_latency(
                node, user_location)  # latency to VNF location
            total_resources += resources
            total_latency += latency
            neighbor_count += 1

    if neighbor_count == 0:
        return 0
    # v(T) = resource / average latency
    return total_resources / (total_latency / neighbor_count)


# Choose Ns for VNF (by x(T))
def embedding_group_with_latency(vnf, MP, substrate_network, user_location):
    # min_latency was a very large number, make it smaller in the process
    min_latency = float('inf')
    best_node = None

    for ns in substrate_network.nodes:
        if ns.can_host(vnf):  # does the node have enough resources?
            # calculate latency
            latency = substrate_network.get_latency(ns, user_location)
            if latency < min_latency:
                min_latency = latency
                best_node = ns

    if best_node is None:
        raise Exception("No suitable substrate node found for VNF")

    # VNF => best substrate node
    best_node.host(vnf)
    vnf.ns = best_node  # best substrate node
    return min_latency  # minimum latency
