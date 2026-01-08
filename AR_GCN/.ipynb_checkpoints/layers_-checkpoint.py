import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(dims[i], dims[i + 1]), nn.GELU()
            ) for i in range(len(dims) - 2)],
            nn.Linear(dims[-2], dims[-1])
        )

    def forward(self, x):
        return self.blocks(x)


class ActNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.LayerNorm(d_model), nn.GELU()
        )

    def forward(self, x):
        return self.blocks(x)

        
class NodeConv(nn.Module):
    def __init__(self, d_model, aggregation):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_model)
        self.W2 = nn.Linear(d_model, d_model)
        self.W3 = nn.Linear(d_model, d_model)
        self.aggregation = aggregation

    def forward(self, x1, x2, e):
        """
        :param x: (b, n, h)
        :param e: (b, n, n, h)
        """
        x1 = self.W1(x1)  # (b, n, h)
        x2 = self.W2(x2)  # (b, n, h)
        e = self.W3(e)
        gates1 = e.sigmoid()  # (b, n, n, h)
        gates2 = gates1.transpose(1, 2)
        x1 = (x1.unsqueeze(1) * gates1).sum(2)
        x2 = (x2.unsqueeze(1) * gates2).sum(2)
        if self.aggregation == 'mean':
            x1 = x1 / (gates1.sum(2) + 1e-10)
            x2 = x2 / (gates2.sum(2) + 1e-10)
        return x1, x2, e, gates1 / (gates1.sum(2, keepdim=True) + 1e-10)


class EdgeConv(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_model)
        self.W2 = nn.Linear(d_model, d_model)
        self.W3 = nn.Linear(d_model, d_model)

    def forward(self, x1, x2, e):
        """
        :param x: (b, n, h)
        :param e: (b, n, n, h)
        """
        e = self.W1(e)  # (b, n, n, h)
        x1 = self.W2(x1).unsqueeze(2)  # (b, 1, n, h)
        x2 = self.W3(x2).unsqueeze(1)  # (b, 1, n, h)
        e = e + x1 + x2
        return e


class GCNLayer(nn.Module):
    def __init__(self, d_model, aggregation):
        super().__init__()
        self.node_conv = NodeConv(d_model, aggregation)
        self.edge_conv = EdgeConv(d_model)
        self.node_norm1 = ActNorm(d_model)
        self.node_norm2 = ActNorm(d_model)
        self.edge_norm = ActNorm(d_model)

    def forward(self, x1, x2, e):
        """
        :param x: (b, n, h)
        :param e: (b, n, n, h)
        """
        e_new = self.edge_conv(x1, x2, e)  # (b, n, n, h)
        e = e + self.edge_norm(e_new)  # (b, n, n, h)
        x1_new, x2_new, logits, scores = self.node_conv(x1, x2, e)  # (b, n, h)
        x1 = x1 + self.node_norm1(x1_new)
        x2 = x2 + self.node_norm2(x2_new)
        return x1, x2, e, logits, scores

