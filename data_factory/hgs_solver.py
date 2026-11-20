"""HGS solutions for symmetric and asymmetric TSP/CVRP instances.

Input under --data_dir:
    --symmetric   : nodes.npy, (b, n, 2) for TSP and (b, n + 1, 2) for CVRP
                    coordinates in the unit square, node 0 is the depot
    (asymmetric)  : edges.npy, (b, n, n) for TSP and (b, n + 1, n + 1) for CVRP
                    explicit distance matrix, node 0 is the depot
    demands.npy   : (b, n) for TSP and (b, n + 1) for CVRP, demands[:, 0] is the
                    vehicle capacity (CVRP only)

Output written when --save_solution is given, matching the names the datasets
load for --baseline hgs (symmetric) and --baseline reference (asymmetric):
    symmetric   : tours_<PROBLEM>_hgs.npy, lengths_<PROBLEM>_hgs.npy
    asymmetric  : tours_<PROBLEM>.npy,     lengths_<PROBLEM>.npy

The scale maps the unit-square instances to integers; hygese rejects distance
matrices far above 1e5 (it aborts instead of raising), so keep
scale * max_distance below 1e5, e.g. 1e4 as used by scripts/hgs_solver_test.sh.
"""
import sys
import argparse
import numpy as np
from time import time
from concurrent.futures import ProcessPoolExecutor

sys.path.append('.')
from AR_GCN.utils import get_length
from hygese import Solver, AlgorithmParameters


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default='CVRP', choices=['TSP', 'CVRP'])
    parser.add_argument('--symmetric', action='store_true')
    parser.add_argument('--vrp_size', type=int)
    parser.add_argument('--data_size', type=int)
    parser.add_argument('--iterations', type=int, default=20000)
    parser.add_argument('--time_factor', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--scale', type=float, default=1e4)
    parser.add_argument('--data_dir', type=str, default='./dataset/SVRP/uniform/100')
    parser.add_argument('--save_solution', action='store_true')
    parser.add_argument('--max_threads', type=int, default=16)
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


def solve(idx, datas, demands):
    n = datas.shape[0]
    if args.symmetric:
        data = {
            'x_coordinates': (datas[:, 0] * args.scale).astype(np.int32),
            'y_coordinates': (datas[:, 1] * args.scale).astype(np.int32),
            'seed': args.seed,
        }
    else:
        data = {
            'distance_matrix': (datas * args.scale).astype(np.int32),
            'seed': args.seed,
        }
    if args.problem == 'CVRP':
        data.update({
            'depot': 0,
            'num_vehicles': n - 1,
            'demands': [0] + demands[1:].tolist(),
            'vehicle_capacity': demands[0].item(),
            'service_times': np.zeros(n),
        })

    param = AlgorithmParameters(nbIter=args.iterations)
    if args.time_factor is not None:
        param.timeLimit = args.time_factor * args.vrp_size

    start_time = time()
    solver = Solver(parameters=param, verbose=False)
    solution = solver.solve_cvrp(data) if args.problem == 'CVRP' else solver.solve_tsp(data)
    end_time = time()
    runtime = end_time - start_time
    tour, length = read_solution(solution)

    if args.problem == 'CVRP' and not args.symmetric:
        # hygese only sees the scaled int32 data; recompute the length on the float data to cancel the quantization error
        length = get_length(datas, tour, flatten=False).item()

    print(f'Solved instance-{idx}')
    print(f'Time: {runtime:.2f}s | Length: {length:.4f}')
    return runtime, tour, length


if __name__ == '__main__':
    args = parse_args()

    Demands = np.load(f'{args.data_dir}/demands.npy')
    if args.symmetric:
        Datas = np.load(f'{args.data_dir}/nodes.npy')
        if args.problem == 'TSP':
            Datas = Datas[:, 1:]
    else:
        Datas = np.load(f'{args.data_dir}/edges.npy')
        if args.problem == 'TSP':
            Datas = Datas[:, 1:, 1:]

    args.data_size = args.data_size or Datas.shape[0]
    args.vrp_size = args.vrp_size or Datas.shape[1] - int(args.problem == 'CVRP')
    with ProcessPoolExecutor(max_workers=min(args.data_size, args.max_threads)) as executor:
        results = executor.map(solve, range(args.data_size), Datas[:args.data_size], Demands[:args.data_size])

    times, tours, lengths = [], [], []
    for result in results:
        times.append(result[0])
        tours.append(result[1])
        lengths.append(result[2])

    times = np.array(times)
    tours = np.array(tours, dtype=np.int32)
    lengths = np.array(lengths, dtype=np.float32)

    avg_time = times.mean()
    avg_length = lengths.mean()

    print(f'Avg Time: {avg_time:.2f}s | Avg Length: {avg_length:.4f}')

    if args.save_solution:
        suffix = '_hgs' if args.symmetric else ''
        tour_file = f'{args.data_dir}/tours_{args.problem}{suffix}.npy'
        length_file = f'{args.data_dir}/lengths_{args.problem}{suffix}.npy'
        np.save(tour_file, tours)
        np.save(length_file, lengths)
        print(f'Saved tours to {tour_file} | Shape: {tours.shape}')
