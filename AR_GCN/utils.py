import sys
import torch
import random
import numpy as np
from scipy.spatial.distance import pdist, squareform


class Logger:
    def __init__(self, log_path, mode='a'):
        self.stdout = sys.stdout  # keep the original stdout (console)
        self.file = open(log_path, mode, encoding='utf-8')  # target log file

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.flush()  # write through immediately so nothing is buffered

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def node_to_edge(nodes):
    upper = pdist(nodes, metric='euclidean').astype(np.float32)
    edges = squareform(upper)  # (n, n)
    return edges


def min_max_clamp(edges, zero=True, only_scale=True):
    if not only_scale:
        # max_others = edges[:, 1:, 1:].amax(dim=[1, 2])  # longest edge excluding the depot
        # edges[..., 0] = torch.min(edges[..., 0], max_others.unsqueeze(1))
        max_others = edges[..., 1:].amax(dim=2)  # longest edge excluding the depot
        edges[..., 0] = torch.min(edges[..., 0], max_others)
        edges[:, 0] = 0.0
    maxi = edges.amax(dim=[1, 2], keepdim=True)
    mini = 0 if zero else edges.amin(dim=[1, 2], keepdim=True)
    edges = (edges - mini) / (maxi - mini + 1e-10)
    return edges


def get_curr_knn(k, edges, cids, masks_hard, masks_soft=None, sorted=False):
    """
    :param k: subgraph size, i.e. the number of candidate nodes
    :param cids: (b) current node ids
    :param dids: (b) destination (depot) node ids
    :param edges: (b, n, n) distance matrix
    :param masks: (b, n) mask, False = selectable, True = excluded
    :param sorted: whether the candidate ids are sorted
    """
    rng = torch.arange(cids.shape[0])
    e = edges[rng, cids]  # (b, n)
    if masks_soft is not None:
        e[masks_soft] += 1e6  # (b, n)
    e[masks_hard] = torch.inf
    _, indices = e.topk(k=k, largest=False, sorted=sorted)
    return indices


def get_curr_demand(demands, remains, indices):
    """
    :param demands: (b, n)
    :param remains: (b)
    :param indices: (b, k)
    :return: (b, k)
    """
    demands = demands.gather(dim=1, index=indices)  # (b, k)
    demands[:, 1] = remains
    return demands


def flatten_cvrp_tour(tour):
    tour_flatten = []
    for x in tour:
        tour_flatten.append(x[0])
        if x[1] == 1:
            tour_flatten.append(0)
    return np.array(tour_flatten) if isinstance(tour, np.ndarray) else \
        torch.tensor(tour_flatten, dtype=tour.dtype, device=tour.device)


def is_valid_tour(tour, capacity=None, demands=None, problem='TSP') -> bool:
    """
    :param tour: Flatten tour of 1-dimension
    """
    tour = torch.as_tensor(tour)
    if tour[0] != tour[-1]:
        return False
    if problem.upper() == 'TSP':
        uniques = tour.unique()
        return len(tour) == len(uniques) + 1

    demands = torch.as_tensor(demands)
    remain = capacity
    for x in tour[1:]:
        if x == 0:
            remain = capacity
            continue
        remain = remain - demands[x]
        if remain < 0:
            return False
    uniques = tour.unique()
    return len(demands) == len(uniques)


def get_length(edges, tours, flatten=True):
    """
    :param edges: (n, n) or (b, n, n)
    :param tours: (n + 1) or (b, n + 1)
    :return: len: float or (b)
    """
    edges = torch.as_tensor(edges)
    tours = torch.as_tensor(tours)
    device = edges.device
    tours = tours.clone()
    if flatten:
        if edges.ndim == 2 and tours.ndim == 1:
            return edges[tours[:-1], tours[1:]].sum()
        elif edges.ndim == 2 and tours.ndim == 2:
            return edges[tours[:, :-1], tours[:, 1:]].sum(-1)
        elif edges.ndim == 3 and tours.ndim == 2 and edges.shape[0] == tours.shape[0]:
            sid = tours[:, :-1]
            eid = tours[:, 1:]
            bid = torch.arange(edges.shape[0], device=device).unsqueeze(1).expand_as(sid)
            return edges[bid, sid, eid].sum(-1)
        else:
            raise ValueError(f"Invalid input shapes. Edges: {edges.shape}, Tours: {tours.shape}")
            
    if edges.ndim == 2 and tours.ndim == 2:
        length = 0
        dest_id = tours[0, 0]
        curr_id = dest_id
        for x in tours[1:]:
            length += edges[curr_id, x[0]]
            curr_id = x[0]
            if x[1] == 1:
                length += edges[curr_id, dest_id]
                curr_id = dest_id
        return length
    elif edges.ndim == 2 and tours.ndim == 3:
        lengths = torch.zeros(tours.shape[0], device=device)
        dest_ids = tours[:, 0, 0]
        curr_ids = dest_ids
        for x in tours[:, 1:].transpose(0, 1):
            lengths += edges[curr_ids, x[:, 0]]
            curr_ids = x[:, 0]
            masks = (x[:, 1] == 1)
            if masks.any():
                lengths[masks] += edges[curr_ids[masks], dest_ids[masks]]
                curr_ids[masks] = dest_ids[masks]
        return lengths
    elif edges.ndim == 3 and tours.ndim == 3 and edges.shape[0] == tours.shape[0]:
        rng = torch.arange(tours.shape[0], device=device)
        lengths = torch.zeros(tours.shape[0], device=device)
        dest_ids = tours[:, 0, 0]
        curr_ids = dest_ids
        for x in tours[:, 1:].transpose(0, 1):
            lengths += edges[rng, curr_ids, x[:, 0]]
            curr_ids = x[:, 0]
            masks = (x[:, 1] == 1)
            if masks.any():
                lengths[masks] += edges[masks, curr_ids[masks], dest_ids[masks]]
                curr_ids[masks] = dest_ids[masks]
        return lengths
    else:
        raise ValueError(f"Invalid input shapes. Edges: {edges.shape}, Tours: {tours.shape}")
    

def get_gap(opt_lens, pred_lens):
    """
    :param opt_lens: float or (b, )
    :param pred_lens: float or (b, )
    :return: gaps: float or (b, )
    """
    return pred_lens / opt_lens - 1
