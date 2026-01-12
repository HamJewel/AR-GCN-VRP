# this file is to generate different distributed SVRP instances with different size including
# size: arbitrary specified
# distribution:
#     uniform: each node is uniformly i.i. distributed
#     clustered: uniformly distributed center with Gaussian distributed nodes around each center
#     explosion: uniformly distributed nodes, but not in
#     implosion: uniformly distributed nodes, but not in
# for every output, it is processed by min-max normalization over a [0,1]^2 unit square
# the output instances are given in torch.Tensor
#
# IMPLEMENTATION NOTES:
#     due to some reasons, instance generation are performed in something like for-loops
#     but this should not impose any inconvenience, since data generation it is not time-consuming
import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.distributions import Exponential
sys.path.append('.')
from AR_GCN.utils import set_seed, node_to_edge


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--vrp_size', type=int, default=1000)
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--min_demand', type=int, default=1)
    parser.add_argument('--max_demand', type=int, default=10)
    parser.add_argument('--data_size', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='./dataset/SVRP_test/')
    parser.add_argument('--svrp_distribution', type=str, choices=['uniform', 'clustered', 'explosion', 'implosion'])
    parser.add_argument('--avrp_distribution', type=str, choices=['tmat', 'perturbation'])
    parser.add_argument('--cluster_num', type=int, default=3)
    parser.add_argument('--cluster_diversity', type=float, default=10)
    parser.add_argument('--min_range', type=float, default=0.1)
    parser.add_argument('--max_range', type=float, default=0.5)
    parser.add_argument('--explosion_rate', type=float, default=10)
    parser.add_argument('--min_tmat', type=int, default=0)
    parser.add_argument('--max_tmat', type=int, default=1000000)
    parser.add_argument('--scale', type=float, default=1e6)
    parser.add_argument('--perturbation_factor', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--symmetric', action='store_true')
    return parser.parse_args()


def normalize_to_unit_square(nodes):
    """
    normalize a svrp instance to a [0, 1]^2 unit square, prefer to have points on both x=0 and y=0
    :param nodes: a (vrp_size, 2) tensor
    :return: a (vrp_size, 2) tensor, a normalized svrp instance
    """
    nodes_norm = nodes.clone()
    maxi = nodes_norm.amax(dim=0)
    mini = nodes_norm.amin(dim=0)
    factor = (maxi - mini).amax()
    nodes_norm = (nodes_norm - mini) / factor
    return nodes_norm


def generate_uniform_svrp():
    """
    vrp_size, min_demand=1, max_demand=10, capacity=50
    """
    points = normalize_to_unit_square(torch.rand((args.vrp_size + 1, 2)))
    demands = torch.randint(args.min_demand, args.max_demand + 1, [args.vrp_size + 1])
    demands[0] = args.capacity
    return points.float().numpy(force=True), demands.int().numpy(force=True)


def generate_clustered_svrp():
    """
    vrp_size, min_demand=1, max_demand=10, capacity=50, cluster_num=3, cluster_diversity=10
    """
    centers = args.cluster_diversity * torch.rand((args.cluster_num + 1, 2))
    nodes = torch.normal(0, 1, (args.vrp_size, 2))
    group = torch.randint(args.cluster_num, [args.vrp_size])
    points = nodes + centers[group]
    points = torch.cat((centers[-1].unsqueeze(dim=0), points), dim=0)
    points = normalize_to_unit_square(points)
    demands = torch.randint(args.min_demand, args.max_demand + 1, [args.vrp_size + 1])
    demands[0] = args.capacity
    return points.float().numpy(force=True), demands.int().numpy(force=True)


def generate_explosion_svrp():
    """
    vrp_size, min_demand=1, max_demand=10, capacity=50, min_range=0.1, max_range=0.5, rate=10
    """
    nodes = torch.rand((args.vrp_size, 2))
    explosion_center = torch.rand([2])
    pointer_vector = nodes - explosion_center
    explosion_range = (args.max_range - args.min_range) * torch.rand(1) + args.min_range
    exploded = pointer_vector.norm(dim=1) < explosion_range
    explosion_factor = explosion_range + Exponential(rate=args.explosion_rate).sample([args.vrp_size, 1])
    directional_vector = pointer_vector / pointer_vector.norm(dim=1).unsqueeze(dim=1)
    explosion_movement = directional_vector * explosion_factor
    nodes[exploded] = explosion_center + explosion_movement[exploded]
    points = torch.cat([torch.rand((1, 2)), nodes], dim=0)
    points = normalize_to_unit_square(points)
    demands = torch.randint(args.min_demand, args.max_demand + 1, [args.vrp_size + 1])
    demands[0] = args.capacity
    return points.float().numpy(force=True), demands.int().numpy(force=True)


def generate_implosion_svrp():
    """
    vrp_size, min_demand=1, max_demand=10, capacity=50, min_range=0.1, max_range=0.5
    """
    nodes = torch.rand((args.vrp_size, 2))
    implosion_center = torch.rand([2])
    pointer_vector = nodes - implosion_center
    implosion_range = (args.max_range - args.min_range) * torch.rand(1) + args.min_range
    imploded = pointer_vector.norm(dim=1) < implosion_range
    implosion_factor = min(implosion_range, torch.normal(0, 1, [1]))
    implosion_movement = pointer_vector * implosion_factor
    nodes[imploded] = implosion_center + implosion_movement[imploded]
    points = torch.cat([torch.rand((1, 2)), nodes], dim=0)
    points = normalize_to_unit_square(points)
    demands = torch.randint(args.min_demand, args.max_demand + 1, [args.vrp_size + 1])
    demands[0] = args.capacity
    return points.float().numpy(force=True), demands.int().numpy(force=True)


def generate_tmat_avrp():
    edges = torch.randint(args.min_tmat, args.max_tmat + 1, (args.vrp_size + 1, args.vrp_size + 1), device=args.device)
    edges.fill_diagonal_(0)
    while True:
        edges_old = edges
        edges = (edges.unsqueeze(1) + edges.unsqueeze(0).transpose(1, 2)).amin(dim=2)
        if torch.equal(edges, edges_old):
            break
    edges = edges / args.scale
    demands = torch.randint(args.min_demand, args.max_demand + 1, [args.vrp_size + 1])
    demands[0] = args.capacity
    return edges.float().numpy(force=True), demands.int().numpy(force=True)


def generate_perturbation_avrp():
    nodes, demands = generate_uniform_svrp()
    edges = node_to_edge(nodes)
    factor = args.perturbation_factor
    random_vals = np.random.uniform(-factor, factor, (args.vrp_size + 1, args.vrp_size + 1))
    upper_tri = np.triu(random_vals, k=1)  # 严格上三角
    perturbation = upper_tri - upper_tri.T
    edges = edges * (1 + perturbation)
    return edges, demands


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    generator = {
        'tmat': generate_tmat_avrp, 'perturbation': generate_perturbation_avrp,
        'uniform': generate_uniform_svrp, 'clustered': generate_clustered_svrp,
        'explosion': generate_explosion_svrp, 'implosion': generate_implosion_svrp
    }[args.svrp_distribution if args.symmetric else args.avrp_distribution]
    data_name = 'nodes' if args.symmetric else 'edges'
    datas, demands = [], []
    for i in tqdm(range(args.data_size)):
        x, y = generator()
        datas.append(x)
        demands.append(y)
    datas = np.stack(datas, axis=0)
    datas = np.round(datas, decimals=6)
    demands = np.stack(demands, axis=0)
    print(f'Shape of {data_name}: {datas.shape}')
    print(f'Shape of demands: {demands.shape}')
    np.save(f'{args.output_dir}/{data_name}.npy', datas)
    np.save(f'{args.output_dir}/demands.npy', demands)
