from math import radians, sin, cos, sqrt, atan2
class SubstrateNetwork:
    def __init__(self):
        self.nodes = []  # all substrate nodes
        self.edges = {}  # all edges

    def get_latency(self, node, location):
        # Calculate Haversine distance as latency (km) from node to user location
        return haversine(node.location[0], node.location[1], location[0], location[1])


class SubstrateNode:
    def __init__(self, id, cpu, location):
        self.id = id
        self.cpu = cpu
        self.location = location
        self.hosted_vnfs = []

    def can_host(self, vnf):
        # if there are enough resources
        return self.cpu >= vnf.cpu_req

    def host(self, vnf):
        self.hosted_vnfs.append(vnf)
        self.cpu -= vnf.cpu_req

    def distance_to(self, location):
        return ((self.location[0] - location[0])**2 + (self.location[1] - location[1])**2)**0.5


class VNF:
    def __init__(self, id, cpu_req, location):
        self.id = id
        self.cpu_req = cpu_req
        self.location = location  # VNF location
        self.ns = None  # substrate node that hosts the VNF

def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Earth radius in km
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c