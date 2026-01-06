import sys
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform


class Logger:
    def __init__(self, log_path, mode='a'):
        self.stdout = sys.stdout  # 记录原始的标准输出（屏幕）
        self.file = open(log_path, mode, encoding='utf-8')  # 打开目标文件

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.flush()  # 确保数据立即被写入，避免延迟

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


def min_max_clamp(edges, zero=True):
    max_others = edges[..., 1:].amax(dim=2)  # 不包含起点的最长边
    edges[..., 0] = torch.min(edges[..., 0], max_others)
    edges[:, 0] = 0.0
    maxi = edges.amax(dim=[1, 2], keepdim=True)
    mini = 0 if zero else edges.amin(dim=[1, 2], keepdim=True)
    edges = (edges - mini) / (maxi - mini + 1e-10)
    return edges


def get_curr_knn(k, edges, cids, masks_hard, masks_soft=None, sorted=False):
    """
    :param k: 子图大小
    :param cids: (b) 当前节点id
    :param dids: (b) 终点节点id
    :param edges: (b, n, n) 距离矩阵
    :param masks: (b, n) 遮罩矩阵, False为可选，True为不可选
    :param sorted: 是否排序
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


def is_valid_tour(tour, demands=None, problem='TSP') -> bool:
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
    remain = demands[0]
    for x in tour[1:]:
        if x == 0:
            remain = demands[0]
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


def plot_route(nodes, edges, opt_tour, pred_tour):
    pred_len = get_length(edges, pred_tour)
    opt_len = get_length(edges, opt_tour)
    gap = get_gap(opt_len, pred_len)

    x, y = nodes[:, 0], nodes[:, 1]
    fig, axs = plt.subplots(1, 2, dpi=125)
    axs[0].set_title('Optimal')
    axs[1].set_title(f'Predicted | Gap = {gap:.6%}')
    axs[0].plot(x[opt_tour], y[opt_tour], c='red' if opt_len < pred_len else 'green', label=str(opt_len))
    axs[1].plot(x[pred_tour], y[pred_tour], c='red' if opt_len > pred_len else 'green', label=str(pred_len))
    axs[0].scatter(x, y), axs[1].scatter(x, y)
    axs[0].legend(), axs[1].legend()
    plt.tight_layout()
    plt.show()


def plot_heatmap(probs, tour, order=True):
    if order:
        probs = probs[tour[:-1]][:, tour[:-1]]
    plt.figure(dpi=125)
    plt.imshow(probs, vmin=0.0, vmax=1.0)
    plt.title('Heatmap of Probabilities')
    plt.colorbar()
    plt.show()
