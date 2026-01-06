import torch
from torch import nn
from AR_GCN.layers_ import MLP, GCNLayer
"""Revisiting Autoregressive GCNs for Vehicle Routing Problems"""


class Embedding(nn.Module):
    def __init__(self, d_model, d_gate, max_k=0, rank_emb=False):
        super().__init__()
        scale = d_model ** 0.5
        k = max(1, max_k) if rank_emb else 1
        self.cand_emb = nn.Parameter(torch.randn(1, k, d_model) / scale)
        self.dest_curr_emb = nn.Parameter(torch.randn(1, 2, d_model) / scale)
        self.demand_emb = nn.Linear(1, d_model)
        self.edge_emb = nn.Linear(1, d_gate)

    def forward(self, edges, demands=None):
        """
        :param edges: (b, n, n, 1)
        :param demands: (b, n, 1)
        """
        b, n = edges.shape[:2]
        cands = self.cand_emb[:, :n - 2].expand(1, n - 2, -1)  # (1, n-2, h)
        x = torch.cat([self.dest_curr_emb, cands], dim=1)  # (1, n, h)
        x = x.expand(b, -1, -1)  # (b, n, h)
        if demands is not None:
            x = x + self.demand_emb(demands)  # (b, n, h)
        e = self.edge_emb(edges)
        return x, e


class TSP_Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding = Embedding(args.d_model, args.d_gate, args.max_k, args.rank_emb)
        self.layers = nn.ModuleList([
            GCNLayer(args.d_model, args.d_gate, args.aggregation) for i in range(args.n_encoders)
        ])
        self.projection = MLP([2 * args.d_model, args.d_model, 1])
        self.criterion = nn.CrossEntropyLoss()
        self.logits = [None] * len(self.layers)
        self.scores = [None] * len(self.layers)
        self.h = args.d_model

    def forward(self, edges):
        """
        :param edges: (b, n, n, 1)
        """
        x, e = self.embedding(edges)  # (b, n, n, h)
        x1 = x2 = x
        for i, layer in enumerate(self.layers):
            x1, x2, e, logits, scores = layer(x1, x2, e)
            self.logits[i] = logits.mean(dim=(0, 1, 2))
            self.scores[i] = scores.mean(dim=(0, 1, 2))
        x = torch.cat([x1, x2], dim=2)
        logits = self.projection(x).squeeze(2)  # (b, n)
        return logits  # (b, n)

    @torch.no_grad()
    def predict(self, edges, indices):
        logits = self.forward(edges)
        logits[:, :2] = -torch.inf
        idx = logits.argmax(dim=1)  # (b)
        x = indices[torch.arange(edges.size(0)), idx]  # (b)
        return x

    def get_loss(self, edges, indices, labels):
        """
        labels: (batch)
        """
        logits = self.forward(edges)  # (b, n)
        logits[:, :2] = -torch.inf
        matches = (indices == labels.unsqueeze(1))  # (b, n)
        targets = matches.int().argmax(dim=1)  # (b)
        masks = (targets == 0)
        targets[masks] = -100
        loss = self.criterion(logits, targets)
        return loss


class CVRP_Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding = Embedding(args.d_model, args.d_gate, args.max_k, args.rank_emb)
        self.layers = nn.ModuleList([
            GCNLayer(args.d_model, args.d_gate, args.aggregation) for i in range(args.n_encoders)
        ])
        self.projection = MLP([2 * args.d_model, args.d_model, 2])
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, edges, demands):
        """
        :param edges: (b, n, n, 1)
        :param demands: (b, n, 1)
        """
        x, e = self.embedding(edges, demands)  # (b, n, n, h)
        x1 = x2 = x
        for layer in self.layers:
            x1, x2, e = layer(x1, x2, e)
        x = torch.cat([x1, x2], dim=2)
        logits = self.projection(x)  # (b, n, 2)
        return logits

    @torch.no_grad()
    def predict(self, edges, demands, indices, masks=None, topk=1):
        logits = self.forward(edges, demands)  # (b, n, 2)
        if masks is None:
            masks = (demands[..., 0] > demands[:, 1])  # (b, n, 1)
            masks[:, :2] = True
        masks = masks.unsqueeze(2).expand(-1, -1, 2)
        logits[masks] = -torch.inf
        _, idx = logits.flatten(1).topk(k=topk)  # (b, topk)
        x = indices.gather(dim=1, index=idx // 2)  # (b, topk)
        y = idx % 2  # (b, topk)
        result = torch.stack([x, y], dim=-1)  # (b, topk, 2)
        return result  # (b, topk, 2)

    def get_loss(self, edges, demands, indices, labels):
        """
        labels: (batch, 2)
        indices: (batch, n)
        """
        logits = self.forward(edges, demands)  # (b, n, 2)
        logits[:, :2] = -torch.inf
        matches = (indices == labels[:, :1])  # (b, n)
        targets = matches.int().argmax(dim=1)  # (b)
        masks = (targets == 0)
        targets = 2 * targets + labels[:, 1]  # (b)
        targets[masks] = -100
        loss = self.criterion(logits.flatten(1), targets)
        return loss
