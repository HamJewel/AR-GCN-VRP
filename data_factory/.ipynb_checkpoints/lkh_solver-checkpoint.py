import os
import re
import sys
import argparse
import subprocess
import numpy as np
from time import time
from concurrent.futures import ProcessPoolExecutor

sys.path.append('.')
from AR_GCN.utils import is_valid_tour, flatten_cvrp_tour


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default='TSP', choices=['TSP', 'CVRP'])
    parser.add_argument('--max_trials', type=int)  # 10000
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--vrp_size', type=int)
    parser.add_argument('--vehicles', type=int)
    parser.add_argument('--data_size', type=int)
    parser.add_argument('--time_factor', type=float)  # 1.0
    parser.add_argument('--max_candidates', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--scale', type=float, default=1e6)
    parser.add_argument('--data_dir', type=str, default='./dataset/SVRP/uniform/100')
    parser.add_argument('--output_dir', type=str, default='./data_factory/output')
    parser.add_argument('--lkh_file', type=str, default='./data_factory/LKH-3.0.13/LKH')
    parser.add_argument('--special', action='store_true')
    parser.add_argument('--symmetric', action='store_true')
    parser.add_argument('--keep_output', action='store_true')
    parser.add_argument('--save_solution', action='store_true')
    parser.add_argument('--max_threads', type=int, default=32)
    return parser.parse_args()


def write_parameter(f, demands, problem_file, solution_file):
    f.write(
        f'PROBLEM_FILE = {problem_file}\n'
        f'SINTEF_SOLUTION_FILE = {solution_file}\n'
        f'SEED = {args.seed}\n'
        f'RUNS = {args.runs}\n'
        f'MAX_CANDIDATES = {args.max_candidates}\n'
        f'MAX_TRIALS = {args.max_trials or args.vrp_size}\n'
    )
    if args.problem == 'CVRP':
        vehicles = int(np.ceil(demands[1:].sum() / demands[0]))
        f.write(f'VEHICLES = {args.vehicles or vehicles}\n')
    if args.special:
        f.write('SPECIAL\n')
    if args.time_factor is not None:
        f.write(f'TIME_LIMIT = {args.time_factor * args.vrp_size}\n')


def write_problem(f, datas, demands):
    f.write(
        f'TYPE : {"" if args.symmetric else "A"}{args.problem}\n'
        f'DIMENSION : {args.vrp_size + int(args.problem == "CVRP")}\n'
    )
    if args.problem == 'CVRP':
        f.write(f'CAPACITY : {demands[0]}\n')
    if args.symmetric:
        f.write(
            'EDGE_WEIGHT_TYPE : EUC_2D\n'
            'NODE_COORD_SECTION\n'
        )
        for i, x in enumerate(datas, start=1):
            f.write(f'{i} {int(x[0] * args.scale)} {int(x[1] * args.scale)}\n')
    else:
        f.write(
            'EDGE_WEIGHT_TYPE : EXPLICIT\n'
            'EDGE_WEIGHT_FORMAT : FULL_MATRIX\n'
            'EDGE_WEIGHT_SECTION\n'
        )
        for x in datas:
            for y in x:
                f.write(f'{int(y * args.scale)} ')
            f.write('\n')
    if args.problem == 'CVRP':
        f.write(
            'DEMAND_SECTION\n'
            '1 0\n'
        )
        for i, x in enumerate(demands[1:], start=2):
            f.write(f'{i} {x}\n')
        f.write(
            'DEPOT_SECTION\n'
            '1\n'
            '-1\n'
        )
    f.write('EOF')


def text_to_route(text):
    route = text.strip().split()[3:]
    return list(map(int, route))


def read_solution(solution_file, log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        text = f.read()
    pattern = r"Cost\.min\s*=\s*(\d+\.?\d*)"
    match = re.search(pattern, text)
    length = float(match.group(1)) / args.scale

    with open(solution_file, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    if args.problem == 'TSP':
        tour = np.zeros([args.vrp_size + 1], dtype=np.int32)
        tour[1:-1] = text_to_route(lines[-1])
        return tour, length

    tour = np.zeros((args.vrp_size + 1, 2), dtype=np.int32)
    i = 1
    for line in lines[lines.index('Solution') + 1:]:
        route = text_to_route(line)
        k = len(route)
        tour[i: i + k, 0] = route
        i += k
        tour[i - 1, 1] = 1
    return tour, length


def solve(idx, datas, demands):
    solution_file = f'{solution_dir}/{idx}.txt'
    problem_file = f'{problem_dir}/{idx}.txt'
    param_file = f'{param_dir}/{idx}.txt'
    log_file = f'{log_dir}/{idx}.txt'
    with open(param_file, 'w', encoding='utf-8') as f:
        write_parameter(f, demands, problem_file, solution_file)
    with open(problem_file, 'w', encoding='utf-8') as f:
        write_problem(f, datas, demands)

    start_time = time()
    with open(log_file, 'w', encoding='utf-8') as f:
        subprocess.run([args.lkh_file, param_file], stdout=f, stderr=subprocess.STDOUT)
    end_time = time()

    runtime = end_time - start_time
    tour, length = read_solution(solution_file, log_file)
    # if args.problem == 'CVRP' and not is_valid_tour(flatten_cvrp_tour(tour), demands, 'CVRP'):
    #     raise ValueError(f'Invalid tour!')
    if not args.keep_output:
        for file in [solution_file, problem_file, param_file, log_file]:
            os.remove(file)
    print(f'Solved instance-{idx}')
    print(f'Time: {runtime:.2f}s | Length: {length:.4f}')
    return runtime, tour, length


if __name__ == '__main__':
    args = parse_args()
    solution_dir = f'{args.output_dir}/solutions'
    problem_dir = f'{args.output_dir}/problems'
    param_dir = f'{args.output_dir}/params'
    log_dir = f'{args.output_dir}/logs'
    for file_dir in [solution_dir, problem_dir, param_dir, log_dir]:
        os.makedirs(file_dir, exist_ok=True)

    data_name = 'nodes' if args.symmetric else 'edges'
    Demands = np.load(f'{args.data_dir}/demands.npy')
    Datas = np.load(f'{args.data_dir}/{data_name}.npy')
    if args.problem == 'TSP':
        Datas = Datas[:, 1:] if args.symmetric else Datas[:, 1:, 1:]

    args.data_size = args.data_size or Datas.shape[0]
    with ProcessPoolExecutor(max_workers=min(args.data_size, args.max_threads)) as executor:
        results = executor.map(solve, range(args.data_size), Datas, Demands)

    times, tours, lengths = [], [], []
    for result in results:
        times.append(result[0])
        tours.append(result[1])
        lengths.append(result[2])

    times = np.array(times)
    tours = np.array(tours, dtype=np.int32)
    lengths = np.array(lengths, dtype=np.float32)
    metric_text = f'Avg Time: {times.mean():.2f}s | Avg Length: {lengths.mean():.4f}'
    print(metric_text)

    if args.save_solution:
        tour_file = f'{args.data_dir}/tours_{args.problem}.npy'
        length_file = f'{args.data_dir}/lengths_{args.problem}.npy'
        np.save(tour_file, tours)
        np.save(length_file, lengths)
        with open(f'{args.data_dir}/metrics_{args.problem}.txt', 'w', encoding='utf-8') as f:
            f.write(metric_text)
        print(f'Saved tours to {tour_file} | Shape: {tours.shape}')
