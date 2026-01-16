import os
import sys
import json
import torch
import argparse
from pprint import pprint
from torch.utils.data import DataLoader
sys.path.append('.')
from AR_GCN.model import TSP_Net, CVRP_Net
from AR_GCN.data import SVRPSet, AVRPSet, STSPLIBSet, ATSPLIBSet, SCVRPLIBSet
from AR_GCN.utils import Logger, set_seed
from tester import Tester


parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=str, choices=['TSP', 'CVRP'])
parser.add_argument('--d_model', type=int)
parser.add_argument('--d_gate', type=int)
parser.add_argument('--n_encoders', type=int)
parser.add_argument('--max_k', type=int)
parser.add_argument('--aggregation', type=str, choices=['sum', 'mean'])
parser.add_argument('--data_dirs', type=str, nargs='+', required=True)
parser.add_argument('--data_sizes', type=int, nargs='+', required=True)
parser.add_argument('--batch_sizes', type=int, nargs='+', required=True)
parser.add_argument('--n_rollouts', type=int, nargs='+', required=True)
parser.add_argument('--output_dir', type=str, default='./output')
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--cfg_path', type=str)
parser.add_argument('--rec_len', type=int, default=100)
parser.add_argument('--rec_num', type=int, default=0)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--threads', type=int, default=16)
parser.add_argument('--symmetric', action='store_true')
parser.add_argument('--rank_emb', action='store_true')
parser.add_argument('--is_lib', action='store_true')
parser.add_argument('--rewrite', action='store_true')
args = parser.parse_args()
if args.cfg_path and os.path.exists(args.cfg_path):
    with open(args.cfg_path) as f:
        cfg = json.load(f)
    for key, value in cfg.items():
        val = getattr(args, key, None)
        if not val and key != 'max_k':
            setattr(args, key, value)

Dataset = {  # (problem, symmetric, is_lib)
    ('TSP', True, False): SVRPSet,
    ('TSP', False, False): AVRPSet,
    ('CVRP', True, False): SVRPSet,
    ('CVRP', False, False): AVRPSet,
    ('TSP', True, True): STSPLIBSet,
    ('TSP', False, True): ATSPLIBSet,
    ('CVRP', True, True): SCVRPLIBSet,
    ('CVRP', False, True): None
}[(args.problem, args.symmetric, args.is_lib)]
model = eval(f'{args.problem}_Net')(args).cuda()
state_dict = torch.load(args.ckpt_path, weights_only=True)
model.load_state_dict(state_dict)
model.eval()
set_seed(args.seed)
tester = Tester(args, model)

os.makedirs(args.output_dir, exist_ok=True)
sys.stderr = sys.stdout = Logger(f'{args.output_dir}/logs_test.txt', 'w' if args.rewrite else 'a')
pprint(vars(args))

if args.is_lib:
    dataset = Dataset(args.data_dirs[0])
    data_size = len(dataset)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=args.threads)
    pred_lens, opt_lens, opt_gaps, times = tester.test(data_loader, args.n_rollouts[0])
    for i in range(data_size):
        print(f'=======[{dataset.instances[i]}]=======')
        print(f'Pred Len:\t{pred_lens[i]:.4f}')
        print(f'Opt Len:\t{opt_lens[i]:.4f}')
        print(f'Opt Gap:\t{opt_gaps[i]:.4%}')
        print(f'Time:\t{times[i]:.4f}s')
        print('=======================')
    avg_time = times.sum() / data_size
    print('=======[Summary]=======')
    print(f'Data Size:\t{data_size}')
    print(f'Avg Pred Len:\t{pred_lens.mean():.4f}')
    print(f'Avg Opt Len:\t{opt_lens.mean():.4f}')
    print(f'Avg Opt Gap:\t{opt_gaps.mean():.4%}')
    print(f'Avg Time:\t{avg_time:.4f}s')
    print('=======================')
    sys.exit(0)

for data_dir, data_size, batch_size, n_rollout in zip(args.data_dirs, args.data_sizes, args.batch_sizes, args.n_rollouts):
    dataset = Dataset(data_dir, data_size, problem=args.problem)
    data_size = len(dataset)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=args.threads)
    pred_lens, opt_lens, opt_gaps, times = tester.test(data_loader, n_rollout)
    avg_time = times.sum() / data_size
    print('=======[Summary]=======')
    print(f'Node Scale:\t{dataset.n}')
    print(f'Data Size:\t{data_size}')
    print(f'Avg Pred Len:\t{pred_lens.mean():.4f}')
    print(f'Avg Opt Len:\t{opt_lens.mean():.4f}')
    print(f'Avg Opt Gap:\t{opt_gaps.mean():.4%}')
    print(f'Avg Time:\t{avg_time:.4f}s')
    print('=======================')
