import os
import torch
import numpy as np
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from AR_GCN.utils import min_max_clamp, get_curr_knn, get_curr_demand


class Trainer:
    def __init__(self, args, model, optimizer, scheduler, writer):
        self.args = args
        self.model = model
        self.device = f'cuda:{args.cuda_id}'
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer
        self.global_steps = 0
        self._process_batch = eval(f'self._step_{args.problem}')

    def _step_CVRP(self, batch):
        edges, demands, tours = batch['edges'].to(self.device), batch['demands'].to(self.device), batch['tours'].to(self.device)
        batch_size, n_nodes = edges.shape[:2]
        rng = torch.arange(batch_size, device=self.device)
        masks = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=self.device)
        dest_ids = tours[:, 0, 0]
        capacities = demands[:, 0].clone()
        remains = demands[:, 0].clone()
        d_glob = demands.clone()
        d_glob[:, 0] = 0

        loss_list, grad_norm_list = [], []
        for step in range(1, n_nodes):
            curr_ids = tours[:, step - 1, 0]
            labels = tours[:, step]
            masks[rng, curr_ids] = True
            masks_depot = (tours[:, step - 1, 1] == 1)
            curr_ids = torch.where(masks_depot, dest_ids, curr_ids)
            remains = torch.where(masks_depot, capacities, remains - d_glob[rng, curr_ids])

            k_nodes = min(self.args.max_k, n_nodes - step) if self.args.max_k > 0 else n_nodes - step
            masks_demand = (d_glob > remains.unsqueeze(1))
            indices = get_curr_knn(k_nodes, edges, curr_ids, masks, masks_demand)
            indices = torch.cat([dest_ids[:, None], curr_ids[:, None], indices], dim=1)

            e_curr = edges[rng[:, None, None], indices[:, :, None], indices[:, None, :]]
            e_curr = min_max_clamp(e_curr, zero=True, only_scale=True)
            augs = 0.5 + 1.0 * torch.rand(batch_size, 1, 1, device=self.device)
            e_curr = e_curr * augs
            d_curr = get_curr_demand(d_glob, remains, indices)
            d_curr = d_curr.float() / capacities.unsqueeze(1)

            self.optimizer.zero_grad()
            loss = self.model.get_loss(e_curr.unsqueeze(3), d_curr.unsqueeze(2), indices, labels)
            loss.backward()
            grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            loss_list.append(loss.item())
            grad_norm_list.append(grad_norm.item())

        return {'loss': loss_list, 'grad_norm': grad_norm_list}

    def _step_TSP(self, batch):
        edges, tours = batch['edges'].to(self.device), batch['tours'].to(self.device)
        batch_size, n_nodes = edges.shape[:2]
        rng = torch.arange(batch_size, device=self.device)
        masks = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=self.device)
        dest_ids = tours[:, 0]

        loss_list, grad_norm_list = [], []
        for step in range(1, n_nodes):
            curr_ids = tours[:, step - 1]
            labels = tours[:, step]
            masks[rng, curr_ids] = True

            k_nodes = min(self.args.max_k, n_nodes - step) if self.args.max_k > 0 else n_nodes - step
            indices = get_curr_knn(k_nodes, edges, curr_ids, masks)
            indices = torch.cat([dest_ids[:, None], curr_ids[:, None], indices], dim=-1)

            e_curr = edges[rng[:, None, None], indices[:, :, None], indices[:, None, :]]
            e_curr = min_max_clamp(e_curr, zero=True, only_scale=not self.args.symmetric)

            self.optimizer.zero_grad()
            loss = self.model.get_loss(e_curr.unsqueeze(3), indices, labels)
            loss.backward()
            grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            loss_list.append(loss.item())
            grad_norm_list.append(grad_norm.item())

        return {'loss': loss_list, 'grad_norm': grad_norm_list}

    def _save_ckpt(self, filename):
        ckpt_path = os.path.join(self.args.ckpt_dir, filename)
        torch.save(self.model.state_dict(), ckpt_path)
        print(f'Checkpoint saved: {ckpt_path}')

    def train(self, data_loader):
        save_steps = max(1, len(data_loader) // 10)
        for epoch in range(1, self.args.n_epoch + 1):
            for batch in tqdm(data_loader, desc=f'Training [Epoch-{epoch}]'):
                self.global_steps += 1
                results = self._process_batch(batch)
                results = {k: np.mean(v).item() for k, v in results.items()}
                self.scheduler.step()
                # print(f'Max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB')
                for k, v in results.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_steps)
                if self.global_steps % self.args.log_steps == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f"step: {self.global_steps} | lr: {lr:.8f}")
                    print(" | ".join([f"{k}: {v:.6f}" for k, v in results.items()]))
                if self.global_steps % save_steps == 0:
                    self._save_ckpt(f'steps-{self.global_steps}.pth')

            self._save_ckpt(f'epoch-{epoch}.pth')
