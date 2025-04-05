class SubstrateNetwork:
    def __init__(self):
        self.nodes = []  # all substrate nodes
        self.edges = {}  # all edges

    def get_latency(self, node, location):
        # Ns => mobile user location
        return node.distance_to(location)


class SubstrateNode:
    def __init__(self, cpu, location):
        self.cpu = cpu
        self.location = location
        self.hosted_vnfs = []

    def can_host(self, vnf):
        # if there are enough resources
        return self.cpu >= vnf.cpu_req and self.memory >= vnf.mem_req

    def host(self, vnf):
        self.hosted_vnfs.append(vnf)
        self.cpu -= vnf.cpu_req

    def distance_to(self, location):
        return ((self.location[0] - location[0])**2 + (self.location[1] - location[1])**2)**0.5


class VNF:
    def __init__(self, cpu_req, mem_req, location):
        self.cpu_req = cpu_req
        self.location = location  # VNF location
        self.ns = None  # substrate node that hosts the VNF
