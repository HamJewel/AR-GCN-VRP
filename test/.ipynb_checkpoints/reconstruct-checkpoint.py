import os
import sys
import json
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pprint import pprint
from torch.utils.data import DataLoader
sys.path.append('.')
from AR_GCN.model import TSP_Net
from AR_GCN.data import SVRPSet, AVRPSet
from AR_GCN.utils import Logger, set_seed, min_max_clamp, get_curr_knn, get_length, get_gap, is_valid_tour


def reconstruct(edges, tours, l_path):
    if l_path < 3:
        return tours
    device = edges.device
    batch_size, n_nodes = edges.shape[:2]
    rng = torch.arange(batch_size, device=device)
    n_path = n_nodes // l_path
    masks = torch.ones(batch_size, n_nodes, dtype=torch.bool, device=device)

    shift = torch.randint(n_nodes, (1,)).item()
    paths = torch.roll(tours[:, :-1], shifts=shift, dims=1)
    tours = torch.cat([paths, paths[:, [0]]], dim=1)

    for global_step in tqdm(range(n_path)):
        i_start = global_step * l_path
        i_end = min(i_start + l_path, n_nodes)
        n_rest = i_end - i_start

        rest_indices = tours[:, i_start:i_end]
        src_false = torch.zeros_like(rest_indices, dtype=torch.bool, device=device)
        masks.scatter_(dim=1, index=rest_indices, src=src_false)

        paths = torch.cat([tours[:, [i_start - 1]], tours[:, i_start:i_end + 1]], dim=1)
        dest_ids = paths[:, -1]
        old_diff = get_length(edges, paths)  # (b)

        for local_step in range(n_rest):
            curr_ids = paths[:, local_step]
            k_nodes = min(args.max_k, n_rest - local_step)
            rest_indices = get_curr_knn(k_nodes, edges, curr_ids, masks, sorted=args.rank_emb)

            indices = torch.cat([dest_ids[:, None], curr_ids[:, None], rest_indices], dim=-1)
            e_curr = edges[rng[:, None, None], indices[:, :, None], indices[:, None, :]]
            e_curr = min_max_clamp(e_curr, zero=True)
            next_ids = model.predict(e_curr.unsqueeze(3), indices)
            paths[:, local_step + 1] = next_ids
            masks[rng, next_ids] = True

        new_diff = get_length(edges, paths)  # (b)
        diff_masks = (old_diff - new_diff > 1e-6)
        if diff_masks.any():
            b_idx = torch.where(diff_masks)[0]
            # print('Successful Reconstruction:', len(b_idx))
            tours[b_idx, i_start:i_end] = paths[b_idx, 1:-1]

    return tours


def parallel_reconstruct(edges, tours, l_path):
    if l_path < 3:
        return tours
    device = edges.device
    B, N = edges.shape[:2]
    # --- 1. Random Roll & Re-cat ---
    paths = tours[:, :-1]
    shift = torch.randint(1, N, (1,)).item()
    paths = paths.roll(shifts=shift, dims=1)
    tours = torch.cat([paths, paths[:, [0]]], dim=1)
    # --- 2. Group Sub-segments (CPU) ---
    groups = {}
    # [关键修复]: 从 1 开始，到 N 结束。
    # 避开索引 0 (Start Anchor) 和 N (End Anchor 同步点)。
    # 这样保证了 start_ids 和 end_ids 永远不会出现在 rest_ids 中。
    for i in range(1, N, l_path):
        # 确保不越界，且不触碰 N (虽然 range(1, N) 本身就不包含 N，但 min 逻辑要稳)
        length = min(i + l_path, N) - i
        if length <= 0:
            continue
        if length not in groups:
            groups[length] = {'s': [], 'e': []}
        groups[length]['s'].append(i)
        groups[length]['e'].append(i + length)
    # --- 3. Batch Inference ---
    flat_tours = tours.view(-1)
    base_offsets = torch.arange(B, device=device) * (N + 1)

    for length, grp in groups.items():
        # A. Indices Preparation
        starts, ends = grp['s'], grp['e']
        n_seg = len(starts)
        tot_sub = B * n_seg
        # 扩展索引
        map_b = torch.arange(B, device=device).repeat_interleave(n_seg)
        # map_b_idx_3d 用于 e_curr 提取: (tot_sub, 1, 1)
        map_b_idx_3d = map_b[:, None, None]

        map_s = torch.tensor(starts, device=device).repeat(B)
        map_e = torch.tensor(ends, device=device).repeat(B)
        # B. Data Extraction
        # 注意: map_s 从 1 开始，偏移量安全
        gather_idx = (base_offsets.repeat_interleave(n_seg)[:, None] +
                      map_s[:, None] +
                      torch.arange(length, device=device)[None, :])

        rest_ids = flat_tours.take(gather_idx)  # (tot_sub, len)
        # Start/End Anchors
        # 因为 map_s >= 1, map_s - 1 >= 0，直接取值即可
        start_ids = tours[map_b, map_s - 1]  # (tot_sub, )
        end_ids = tours[map_b, map_e]  # (tot_sub, )
        # C. Init Paths & Masks
        curr_paths = torch.cat([start_ids[:, None], rest_ids], dim=1)
        dest_expanded = end_ids[:, None]
        # Mask 初始化：1 (Blocked)
        sub_masks = torch.ones(tot_sub, N, dtype=torch.bool, device=device)
        # 将片段内的节点设为 0 (Available)
        sub_masks.scatter_(1, rest_ids, False)
        # 由于我们跳过了 index 0，start_ids (Pre-node) 绝对不在 rest_ids (Segment) 中。
        # 所以 sub_masks 对应的 start_ids 位置本来就是 True (Blocked)。无需额外操作。
        rng_sub = torch.arange(tot_sub, device=device)
        # D. Calc Old Length
        full_old = torch.cat([curr_paths, dest_expanded], dim=1)
        old_diff = edges[map_b[:, None], full_old[:, :-1], full_old[:, 1:]].sum(1)
        # E. Autoregressive Rollout
        for step in range(length):
            curr_ids = curr_paths[:, step]
            k = min(args.max_k, length - step)
            # [显存优化]: Inline KNN
            dists = edges[map_b, curr_ids]
            # TopK (传入 dists 和 sub_masks)
            knn_idx = get_curr_knn(k, dists, sub_masks, curr_ids, sorted=args.rank_emb)
            # --- 构建子图 ---
            indices = torch.cat([dest_expanded, curr_ids[:, None], knn_idx], dim=-1)
            # 提取 e_curr
            e_curr = edges[map_b_idx_3d, indices[:, :, None], indices[:, None, :]]
            e_curr = min_max_clamp(e_curr, zero=True)

            next_ids = model.predict(e_curr.unsqueeze(3), indices)

            curr_paths[:, step + 1] = next_ids
            sub_masks[rng_sub, next_ids] = True  # Block used node
        # F. Evaluation & Scatter Back
        full_new = torch.cat([curr_paths, dest_expanded], dim=1)
        new_diff = edges[map_b[:, None], full_new[:, :-1], full_new[:, 1:]].sum(1)

        diff_mask = (old_diff - new_diff > 1e-6)
        if diff_mask.any():
            up_idx = torch.where(diff_mask)[0]
            # 计算写入索引
            target_gather = (base_offsets[map_b[up_idx]][:, None] +
                             map_s[up_idx][:, None] +
                             torch.arange(length, device=device)[None, :])

            flat_tours.scatter_(0, target_gather.view(-1), curr_paths[up_idx, 1:].view(-1))

    return tours