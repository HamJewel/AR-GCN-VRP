import os
import torch
import numpy as np
from torch.utils.data import Dataset
from AR_GCN.utils import node_to_edge


class SVRPSet(Dataset):
    def __init__(self, data_dir, size=-1, problem='TSP'):
        super().__init__()
        self.nodes = np.load(os.path.join(data_dir, 'nodes.npy'))  # (b, n + 1, 2)
        self.size = size if size > 0 else self.nodes.shape[0]
        self.n = self.nodes.shape[1]
        if problem == 'TSP':
            self.n = self.n - 1
            self.nodes = self.nodes[:, 1:]
        p_tour = os.path.join(data_dir, f'tours_{problem}.npy')
        p_demand = os.path.join(data_dir, 'demands.npy')
        p_length = os.path.join(data_dir, f'lengths_{problem}.npy')
        self.tours = np.load(p_tour) if os.path.exists(p_tour) else np.zeros(self.size, dtype=bool)  # (b, n + 1) or (b, n, 2)
        self.demands = np.load(p_demand) if os.path.exists(p_demand) else np.zeros(self.size, dtype=bool)  # (b, n + 1)
        self.lengths = np.load(p_length) if os.path.exists(p_length) else np.ones(self.size, dtype=bool)  # (b, n + 1)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        edge = node_to_edge(self.nodes[idx]).astype(np.float32)
        demand = self.demands[idx].astype(np.int64)
        tour = self.tours[idx].astype(np.int64)
        length = self.lengths[idx].astype(np.float32)
        # return edge, demand, tour, length
        return {'edges': edge, 'demands': demand, 'tours': tour, 'lengths': length}


class AVRPSet(Dataset):
    def __init__(self, data_dir, size=-1, problem='TSP'):
        super().__init__()
        self.edges = np.load(os.path.join(data_dir, 'edges.npy'))  # (b, n + 1, n + 1)
        self.size = size if size > 0 else self.edges.shape[0]
        self.n = self.edges.shape[1]
        if problem == 'TSP':
            self.n = self.n - 1
            self.edges = self.edges[:, 1:, 1:]
        p_tour = os.path.join(data_dir, f'tours_{problem}.npy')
        p_demand = os.path.join(data_dir, 'demands.npy')
        p_length = os.path.join(data_dir, f'lengths_{problem}.npy')
        self.tours = np.load(p_tour) if os.path.exists(p_tour) else np.zeros(self.size, dtype=bool)  # (b, n + 1) or (b, n, 2)
        self.demands = np.load(p_demand) if os.path.exists(p_demand) else np.zeros(self.size, dtype=bool)  # (b, n + 1)
        self.lengths = np.load(p_length) if os.path.exists(p_length) else np.ones(self.size, dtype=bool)  # (b, n + 1)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        edge = self.edges[idx].astype(np.float32)
        demand = self.demands[idx].astype(np.int64)
        tour = self.tours[idx].astype(np.int64)
        length = self.lengths[idx].astype(np.float32)
        # return edge, demand, tour, length
        return {'edges': edge, 'demands': demand, 'tours': tour, 'lengths': length}


class STSPLIBSet(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.instances = [
            "a280", "berlin52", "bier127", "brd14051", "ch130", "ch150", "d198", "d493",
            "d657", "d1291", "d1655", "d2103", "d15112", "d18512", "eil51", "eil76",
            "eil101", "fl417", "fl1400", "fl1577", "fl3795", "fnl4461", "gil262",
            "kroA100", "kroB100", "kroC100", "kroD100", "kroE100", "kroA150", "kroB150",
            "kroA200", "kroB200", "lin105", "lin318", "nrw1379", "p654",
            "pcb442", "pcb1173", "pcb3038", "pr76", "pr107", "pr124", "pr136", "pr144",
            "pr152", "pr226", "pr264", "pr299", "pr439", "pr1002", "pr2392", "rat99",
            "rat195", "rat575", "rat783", "rd100", "rd400", "rl1304", "rl1323", "rl1889",
            "rl5915", "rl5934", "rl11849", "st70", "ts225", "tsp225", "u159", "u574",
            "u724", "u1060", "u1432", "u1817", "u2152", "u2319", "usa13509", "vm1084", "vm1748"
        ]  # "linhp318" 存在FIXED_EDGES
        self.nodes = [np.load(f'{data_dir}/{x}/nodes.npy') for x in self.instances]  # (b, n, 2)
        self.lengths = [np.load(f'{data_dir}/{x}/length.npy') for x in self.instances]
        self.size = len(self.instances)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        edge = node_to_edge(self.nodes[idx]).astype(np.float32)
        length = self.lengths[idx].astype(np.float32)
        # return np.round(edge), length
        return {'edges': np.round(edge), 'lengths': length}


class ATSPLIBSet(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.instances = [
            "br17", "ft53", "ft70", "ftv33", "ftv35", "ftv38", "ftv44",
            "ftv47", "ftv55", "ftv64", "ftv70", "ftv90", "ftv100", "ftv110",
            "ftv120", "ftv130", "ftv140", "ftv150", "ftv160", "ftv170",
            "kro124p", "p43", "rbg323", "rbg358", "rbg403", "rbg443", "ry48p"
        ]
        self.edges = [np.load(f'{data_dir}/{x}/edges.npy') for x in self.instances]  # (b, n, n)
        self.lengths = [np.load(f'{data_dir}/{x}/length.npy') for x in self.instances]
        self.size = len(self.instances)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        edge = self.edges[idx].astype(np.float32)
        length = self.lengths[idx].astype(np.float32)
        # return np.round(edge), length
        return {'edges': np.round(edge), 'lengths': length}


class SCVRPLIBSet(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.instances = [
            'X-n101-k25', 'X-n106-k14', 'X-n110-k13', 'X-n115-k10', 'X-n120-k6',
            'X-n125-k30', 'X-n129-k18', 'X-n134-k13', 'X-n139-k10', 'X-n143-k7',
            'X-n148-k46', 'X-n153-k22', 'X-n157-k13', 'X-n162-k11', 'X-n167-k10',
            'X-n172-k51', 'X-n176-k26', 'X-n181-k23', 'X-n186-k15', 'X-n190-k8',
            'X-n195-k51', 'X-n200-k36', 'X-n204-k19', 'X-n209-k16', 'X-n214-k11',
            'X-n219-k73', 'X-n223-k34', 'X-n228-k23', 'X-n233-k16', 'X-n237-k14',
            'X-n242-k48', 'X-n247-k50', 'X-n251-k28', 'X-n256-k16', 'X-n261-k13',
            'X-n266-k58', 'X-n270-k35', 'X-n275-k28', 'X-n280-k17', 'X-n284-k15',
            'X-n289-k60', 'X-n294-k50', 'X-n298-k31', 'X-n303-k21', 'X-n308-k13',
            'X-n313-k71', 'X-n317-k53', 'X-n322-k28', 'X-n327-k20', 'X-n331-k15',
            'X-n336-k84', 'X-n344-k43', 'X-n351-k40', 'X-n359-k29', 'X-n367-k17',
            'X-n376-k94', 'X-n384-k52', 'X-n393-k38', 'X-n401-k29', 'X-n411-k19',
            'X-n420-k130', 'X-n429-k61', 'X-n439-k37', 'X-n449-k29', 'X-n459-k26',
            'X-n469-k138', 'X-n480-k70', 'X-n491-k59', 'X-n502-k39', 'X-n513-k21',
            'X-n524-k153', 'X-n536-k96', 'X-n548-k50', 'X-n561-k42', 'X-n573-k30',
            'X-n586-k159', 'X-n599-k92', 'X-n613-k62', 'X-n627-k43', 'X-n641-k35',
            'X-n655-k131', 'X-n670-k130', 'X-n685-k75', 'X-n701-k44', 'X-n716-k35',
            'X-n733-k159', 'X-n749-k98', 'X-n766-k71', 'X-n783-k48', 'X-n801-k40',
            'X-n819-k171', 'X-n837-k142', 'X-n856-k95', 'X-n876-k59', 'X-n895-k37',
            'X-n916-k207', 'X-n936-k151', 'X-n957-k87', 'X-n979-k58', 'X-n1001-k43'
        ]
        self.nodes = [np.load(f'{data_dir}/{x}/nodes.npy') for x in self.instances]  # (b, n, 2)
        self.demands = [np.load(f'{data_dir}/{x}/demands.npy') for x in self.instances]
        self.lengths = [np.load(f'{data_dir}/{x}/length.npy') for x in self.instances]
        self.size = len(self.instances)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        edge = node_to_edge(self.nodes[idx]).astype(np.float32)
        demand = self.demands[idx].astype(np.int64)
        length = self.lengths[idx].astype(np.float32)
        # return np.round(edge), demand, length
        return {'edges': np.round(edge), 'demands': demand, 'lengths': length}
