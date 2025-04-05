# GCBA
def group_connectivity_based_algorithm(substrate_network, vnf_set):
    MP = set()  # 1. Initialize an empty set for mappings

    # 2. Descending-order list of VNF clusters (size ⬇️)
    LV = sorted(vnf_set, key=lambda cluster: len(cluster), reverse=True)

    while LV:  # check every cluster
        Nu_i = LV.pop(0)  # Get the first cluster in LV

        for vnf in Nu_i:  # For each VNF in the cluster
            # Embed the VNF to the appropriate substrate node(Ns)
            embedding_group(vnf, MP)

    return MP  # Return the final mapping of VNFs to substrate nodes


# GBA
def group_based_algorithm(substrate_network, vnf_set):
    MP = set()  # 1. Initialize an empty set for mappings
    # 2. Calculate v(T) for each VNF v in the set of VNFs (neighborhood resource)
    vnf_values = {}  # v(T)
    for vnf in vnf_set:
        vnf_values[vnf] = calculate_vnf_value(vnf)

    # 3. Descending-order list of VNFs based on v(T) value (V(T) ⬇️)
    sorted_vnfs = sorted(
        vnf_set, key=lambda vnf: vnf_values[vnf], reverse=True)
    for vnf in sorted_vnfs:  # Process VNFs one by one in sorted order
        # Step 7: Embed the VNF to the appropriate substrate node
        embedding_group(vnf, MP)

    return MP  # Return the final mapping of VNFs to substrate nodes


# Calculate neighborhood resource (v(T))
def calculate_vnf_value(vnf):
    return 1


# Choose Ns for VNF (by x(T))
def embedding_group(vnf, MP):
    pass
