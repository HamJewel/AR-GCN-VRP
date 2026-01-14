import os
import torch
import numpy as np
from time import time
from tqdm import tqdm
from AR_GCN.utils import min_max_clamp, get_curr_knn, get_curr_demand, get_length, get_gap, is_valid_tour, flatten_cvrp_tour


class Tester:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self._rollout_batch = eval(f'self._rollout_{args.problem}')

    def _rollout_CVRP(self, batch, n_rollout):
        edges, demands = batch['edges'].cuda(), batch['demands'].cuda()
        device = edges.device
        batch_size, n_nodes = edges.shape[:2]
        n_roll = min(self.args.max_k if self.args.max_k > 0 else n_nodes - 1, n_rollout)
        total_size = batch_size * n_roll
        rng1 = torch.arange(batch_size, device=device)
        rng2 = torch.arange(total_size, device=device)
        masks = torch.zeros(total_size, n_nodes, dtype=torch.bool, device=device)
        remains = torch.ones(total_size, dtype=torch.float32, device=device)
        pred_tours = torch.zeros(total_size, n_nodes, 2, dtype=torch.long, device=device)
        dest_ids = torch.zeros(total_size, dtype=torch.long, device=device)
        d_norm = demands / demands[:, [0]]
        d_norm[:, 0] = 0.0

        step = 1
        k_nodes = min(self.args.max_k, n_nodes - step) if self.args.max_k > 0 else n_nodes - step
        masks[:, 0] = True
        indices = get_curr_knn(k_nodes, edges, dest_ids[:batch_size], masks[:batch_size], sorted=self.args.rank_emb)
        indices = torch.cat([torch.zeros(batch_size, 2, dtype=torch.long, device=device), indices], dim=1)
        d_curr = get_curr_demand(d_norm, 1.0, indices)
        masks_curr = masks.gather(dim=1, index=indices)

        e_curr = edges[rng1[:, None, None], indices[:, :, None], indices[:, None, :]]
        e_curr = min_max_clamp(e_curr, zero=True)
        next_ids = self.model.predict(e_curr.unsqueeze(3), d_curr.unsqueeze(2), indices, masks_curr, n_roll)
        pred_tours[:, step] = next_ids.view(-1, 2)
        step += 1

        d_norm = d_norm.repeat_interleave(n_roll, dim=0)
        edges = edges.repeat_interleave(n_roll, dim=0)
        while step < n_nodes:
            masks_depot = (pred_tours[:, step - 1, 1] == 1)
            curr_ids = pred_tours[:, step - 1, 0]
            masks[rng2, curr_ids] = True
            curr_ids = torch.where(masks_depot, dest_ids, curr_ids)
            remains_new = torch.where(masks_depot, 1.0, remains - d_norm[rng2, curr_ids])
            if (remains_new < 0).any():
                raise ValueError(f'Invalid Remains')

            masks_demand = (d_norm > remains_new.unsqueeze(1))
            k_nodes = min(self.args.max_k, n_nodes - step) if self.args.max_k > 0 else n_nodes - step
            indices = get_curr_knn(k_nodes, edges, curr_ids, masks, masks_demand, sorted=self.args.rank_emb)
            indices = torch.cat([dest_ids[:, None], curr_ids[:, None], indices], dim=1)
            d_curr = get_curr_demand(d_norm, remains_new, indices)

            masks_all = (masks | masks_demand)
            masks_curr = masks_all.gather(dim=1, index=indices)
            masks_bid = masks_curr.all(dim=1)
            if masks_bid.any():
                pred_tours[masks_bid, step - 1, 1] = 1
                continue

            remains = remains_new
            e_curr = edges[rng2[:, None, None], indices[:, :, None], indices[:, None, :]]
            e_curr = min_max_clamp(e_curr, zero=True)
            next_ids = self.model.predict(e_curr.unsqueeze(3), d_curr.unsqueeze(2), indices, masks_curr)
            pred_tours[:, step] = next_ids[:, 0]
            step += 1

        pred_tours[:, -1, 1] = 1
        pred_lens = get_length(edges, pred_tours, flatten=False)
        pred_tours = pred_tours.view(batch_size, n_roll, -1, 2)
        pred_lens = pred_lens.view(batch_size, n_roll)
        idx = pred_lens.argmin(dim=1)
        return pred_lens[rng1, idx], pred_tours[rng1, idx]

    def _rollout_TSP(self, batch, n_rollout):
        edges = batch['edges'].cuda()
        device = edges.device
        batch_size, n_nodes = edges.shape[:2]
        n_roll = min(n_nodes, n_rollout)
        total_size = batch_size * n_roll
        edges = edges.repeat_interleave(n_roll, dim=0)
        rng1 = torch.arange(batch_size, device=device)
        rng2 = torch.arange(total_size, device=device)
        dest_ids = torch.arange(n_roll, device=device).repeat(batch_size)
        masks = torch.zeros(total_size, n_nodes, dtype=torch.bool, device=device)
        pred_tours = torch.zeros(total_size, n_nodes + 1, dtype=torch.long, device=device)
        pred_tours[:, 0] = pred_tours[:, -1] = dest_ids

        step = 1
        while step < n_nodes:
            curr_ids = pred_tours[:, step - 1]
            masks[rng2, curr_ids] = True
            k_nodes = min(self.args.max_k, n_nodes - step) if self.args.max_k > 0 else n_nodes - step
            indices = get_curr_knn(k_nodes, edges, curr_ids, masks, sorted=self.args.rank_emb)
            indices = torch.cat([dest_ids[:, None], curr_ids[:, None], indices], dim=-1)
            e_curr = edges[rng2[:, None, None], indices[:, :, None], indices[:, None, :]]
            e_curr = min_max_clamp(e_curr, zero=True)
            next_ids = self.model.predict(e_curr.unsqueeze(3), indices)
            pred_tours[:, step] = next_ids
            step += 1

        pred_lens = get_length(edges, pred_tours, flatten=True)
        pred_lens = pred_lens.view(batch_size, n_roll)
        pred_tours = pred_tours.view(batch_size, n_roll, -1)
        idx = pred_lens.argmin(dim=1)  # (batch_size)
        return pred_lens[rng1, idx], pred_tours[rng1, idx]

    @torch.no_grad()
    def test(self, data_loader, n_rollout):
        all_pred_lens, all_opt_lens, all_opt_gaps, all_times = [], [], [], []

        for bid, batch in tqdm(enumerate(data_loader), desc='Testing'):
            opt_lens = batch['lengths'].cuda()
            start_time = time()
            pred_lens, pred_tours = self._rollout_batch(batch, n_rollout)
            end_time = time()
            opt_gaps = get_gap(opt_lens, pred_lens)

            all_pred_lens.append(pred_lens)
            all_opt_lens.append(opt_lens)
            all_opt_gaps.append(opt_gaps)
            all_times.append(end_time - start_time)

        all_pred_lens = torch.cat(all_pred_lens, dim=0)
        all_opt_lens = torch.cat(all_opt_lens, dim=0)
        all_opt_gaps = torch.cat(all_opt_gaps, dim=0)
        all_times = torch.tensor(all_times)

        return all_pred_lens, all_opt_lens, all_opt_gaps, all_times
