import os
import sys
import argparse
import subprocess
import numpy as np
from time import time
from hygese import Solver, AlgorithmParameters
sys.path.append('.')
from AR_GCN.utils import is_valid_tour, flatten_cvrp_tour


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default='CVRP', choices=['TSP', 'CVRP'])
    parser.add_argument('--vrp_size', type=int)
    parser.add_argument('--data_size', type=int)
    parser.add_argument('--iterations', type=int, default=20000)
    parser.add_argument('--time_factor', type=float)  # 1.0
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--scale', type=float, default=1e6)
    parser.add_argument('--data_dir', type=str, default='./dataset/SVRP/uniform/100')
    parser.add_argument('--save_solution', action='store_true')
    parser.add_argument('--max_threads', type=int, default=32)
    return parser.parse_args()


def read_solution(solution):
    length = solution.cost / args.scale
    tour = np.zeros((args.vrp_size + 1, 2), dtype=np.int32)
    i = 1
    for route in solution.routes:
        k = len(route)
        tour[i: i + k, 0] = route
        i += k
        tour[i - 1, 1] = 1
    if args.problem == 'TSP':
        tour = tour[:, 0]
    return tour, length


def solve(nodes, demands):
    data = {
        'x_coordinates': (nodes[:, 0] * args.scale).astype(np.int32),
        'y_coordinates': (nodes[:, 1] * args.scale).astype(np.int32),
        'seed': args.seed,
    }
    if args.problem == 'CVRP':
        data.update({
            'depot': 0,
            'num_vehicles': args.vrp_size,
            'demands': [0] + demands[1:].tolist(),
            'vehicle_capacity': demands[0].item(),
            'service_times': np.zeros(args.vrp_size + 1)
        })
    # TODO default timelimit for makeup evaluation when the first trial exceed a total time limit
    param = AlgorithmParameters(nbIter=args.iterations)
    if args.time_factor is not None:
        param.timeLimit = args.time_factor * args.vrp_size

    start_time = time()
    solver = Solver(parameters=param, verbose=False)
    solution = solver.solve_cvrp(data) if args.problem == 'CVRP' else solver.solve_tsp(data)
    end_time = time()
    runtime = end_time - start_time
    tour, length = read_solution(solution)
    return runtime, tour, length


if __name__ == '__main__':
    args = parse_args()
    data = dict()
    data['distance_matrix'] = [
        [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
        [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
        [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
        [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
        [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
        [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
        [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
        [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
        [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
        [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
        [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
        [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
        [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0],
    ]

    # Solver initialization
    ap = AlgorithmParameters()  # seconds

    # solve TSP using both coordinates and dist_mtx
    hgs_solver = Solver(parameters=ap, verbose=True)
    result = hgs_solver.solve_tsp(data)
    print(result.cost)
    print(result.routes)
