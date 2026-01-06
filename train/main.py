import os
import sys
import json
import argparse
from os.path import join
from pprint import pprint
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
sys.path.append('.')
from AR_GCN.model import TSP_Net, CVRP_Net
from AR_GCN.data import SVRPSet, AVRPSet
from AR_GCN.utils import Logger, set_seed
from trainer import Trainer


parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=str, choices=['TSP', 'CVRP'])
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--d_gate', type=int, default=16)
parser.add_argument('--n_encoders', type=int, default=6)
parser.add_argument('--max_k', type=int, default=50)
parser.add_argument('--aggregation', type=str, default='mean', choices=['sum', 'mean'])
parser.add_argument('--n_epoch', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--data_dir', type=str, default='./dataset/SVRP_train/uniform/100')
parser.add_argument('--data_size', type=int)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--log_steps', type=int, default=5)
parser.add_argument('--output_dir', type=str, default='./output')
parser.add_argument('--save_name', type=str)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--symmetric', action='store_true')
parser.add_argument('--rank_emb', action='store_true')
parser.add_argument('--rewrite', action='store_true')
args = parser.parse_args()

problem = f'{"S" if args.symmetric else "A"}{args.problem}'
if args.save_name is None:
    args.save_name = f'{problem}-{args.d_model}-{args.d_gate}-{args.n_encoders}-{args.max_k}-{args.n_epoch}-{args.aggregation}'
args.save_dir = join(args.output_dir, args.save_name)
args.ckpt_dir = join(args.save_dir, 'ckpts')
os.makedirs(args.ckpt_dir, exist_ok=True)

writer = SummaryWriter(join('tf-logs', args.save_name))
with open(join(args.save_dir, 'configs.json'), 'w', encoding='utf-8') as f:
    json.dump(vars(args), f, ensure_ascii=False, indent=4)

set_seed(args.seed)
Dataset = SVRPSet if args.symmetric else AVRPSet
dataset = Dataset(args.data_dir, args.data_size, args.problem)
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
total_steps = len(data_loader) * args.n_epoch

model = eval(f'{args.problem}_Net')(args).cuda()
optimizer = Adam(params=model.parameters(), lr=args.learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0.)

sys.stderr = sys.stdout = Logger(join(args.save_dir, 'logs_train.txt'), 'w' if args.rewrite else 'a')
pprint(vars(args))
model.train()
trainer = Trainer(args, model, optimizer, scheduler, writer)
trainer.train(data_loader)
