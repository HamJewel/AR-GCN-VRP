import os
import torch
import numpy as np
from tqdm import tqdm
from AR_GCN.utils import min_max_clamp, get_curr_knn, get_curr_demand


class Trainer:
    def __init__(self, args, model, optimizer, scheduler, writer):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer
        self.global_steps = 0
        self._process_batch = eval(f'self._step_{args.problem}')

    def _step_CVRP(self, batch):
        edges, demands, tours = batch['edges'].cuda(), batch['demands'].cuda(), batch['tours'].cuda()
        batch_size, n_nodes = edges.shape[:2]
        device = edges.device
        rng = torch.arange(batch_size, device=device)
        masks = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=device)
        dest_ids = tours[:, 0, 0]
        remains = torch.ones(batch_size, dtype=torch.float32, device=edges.device)
        d_norm = demands / demands[:, [0]]
        d_norm[:, 0] = 0

        loss_list = []
        for step in range(1, n_nodes):
            curr_ids = tours[:, step - 1, 0]
            labels = tours[:, step]
            masks[rng, curr_ids] = True
            depot_masks = (tours[:, step - 1, 1] == 1)
            curr_ids = torch.where(depot_masks, dest_ids, curr_ids)

            remains = torch.where(depot_masks, 1.0, remains - d_norm[rng, curr_ids])
            masks_demand = (d_norm > remains.unsqueeze(1))

            k_nodes = min(self.args.max_k, n_nodes - step) if self.args.max_k > 0 else n_nodes - step
            indices = get_curr_knn(k_nodes, edges, curr_ids, masks, masks_demand, sorted=self.args.rank_emb)
            indices = torch.cat([dest_ids[:, None], curr_ids[:, None], indices], dim=1)

            e_curr = edges[rng[:, None, None], indices[:, :, None], indices[:, None, :]]
            e_curr = min_max_clamp(e_curr, zero=True).unsqueeze(3)
            d_curr = get_curr_demand(d_norm, remains, indices).unsqueeze(2)

            self.optimizer.zero_grad()
            loss = self.model.get_loss(e_curr, d_curr, indices, labels)
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.item())

        return loss_list

    def _step_TSP(self, batch):
        edges, tours = batch['edges'].cuda(), batch['tours'].cuda()
        batch_size, n_nodes = edges.shape[:2]
        device = edges.device
        rng = torch.arange(batch_size, device=device)
        masks = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=device)
        dest_ids = tours[:, 0]

        loss_list = []
        for step in range(1, n_nodes):
            curr_ids = tours[:, step - 1]
            labels = tours[:, step]
            masks[rng, curr_ids] = True

            k_nodes = min(self.args.max_k, n_nodes - step) if self.args.max_k > 0 else n_nodes - step
            indices = get_curr_knn(k_nodes, edges, curr_ids, masks, sorted=self.args.rank_emb)
            indices = torch.cat([dest_ids[:, None], curr_ids[:, None], indices], dim=-1)

            e_curr = edges[rng[:, None, None], indices[:, :, None], indices[:, None, :]]
            e_curr = min_max_clamp(e_curr, zero=True).unsqueeze(3)

            self.optimizer.zero_grad()
            loss = self.model.get_loss(e_curr, indices, labels)
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.item())

        return loss_list

    def _save_ckpt(self, filename):
        ckpt_path = os.path.join(self.args.ckpt_dir, filename)
        torch.save(self.model.state_dict(), ckpt_path)
        print(f'Checkpoint saved: {ckpt_path}')

    def train(self, data_loader):
        save_steps = max(1, len(data_loader) // 10)
        for epoch in range(1, self.args.n_epoch + 1):
            for batch in tqdm(data_loader, desc=f'Training [Epoch-{epoch}]'):
                self.global_steps += 1
                loss_list = self._process_batch(batch)
                # print(f'Max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB')
                self.scheduler.step()
                loss_item = np.mean(loss_list).item()
                self.writer.add_scalar('train/loss', loss_item, self.global_steps)

                if self.global_steps % self.args.log_steps == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f'Step: {self.global_steps} | Lr: {lr:.8f}\nLoss: {loss_item:.8f}')

                if self.global_steps % save_steps == 0:
                    self._save_ckpt(f'steps-{self.global_steps}.pth')

            self._save_ckpt(f'epoch-{epoch}.pth')
