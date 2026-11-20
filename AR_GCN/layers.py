import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(dims[i], dims[i + 1], bias=False), nn.GELU()
            ) for i in range(len(dims) - 2)],
            nn.Linear(dims[-2], dims[-1], bias=False)
        )

    def forward(self, x):
        return self.blocks(x)


class NodeConv(nn.Module):
    def __init__(self, d_model, aggregation, eps=1e-6):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_model, bias=False)
        self.W2 = nn.Linear(d_model, d_model, bias=False)
        self.W3 = nn.Linear(d_model, d_model, bias=False)
        self.aggregation = aggregation
        self.eps = eps

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
            x1 = x1 / (gates1.sum(2) + self.eps)
            x2 = x2 / (gates2.sum(2) + self.eps)
        return x1, x2


class EdgeConv(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_model, bias=False)
        self.W2 = nn.Linear(d_model, d_model, bias=False)
        self.W3 = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x1, x2, e):
        """
        :param x: (b, n, h)
        :param e: (b, n, n, h)
        """
        e = self.W1(e)  # (b, n, n, h)
        x1 = self.W2(x1).unsqueeze(2)  # (b, n, 1, h)
        x2 = self.W3(x2).unsqueeze(1)  # (b, 1, n, h)
        e = (x1 + x2) * e.sigmoid()
        return e


class GCNLayer(nn.Module):
    def __init__(self, d_model, aggregation):
        super().__init__()
        self.node_conv = NodeConv(d_model, aggregation)
        self.edge_conv = EdgeConv(d_model)
        self.node_norm1 = nn.LayerNorm(d_model)
        self.node_norm2 = nn.LayerNorm(d_model)
        self.edge_norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2, e):
        """
        :param x: (b, n, h)
        :param e: (b, n, n, h)
        """
        e_new = self.edge_conv(x1, x2, e)  # (b, n, n, h)
        e = e + self.edge_norm(e_new)  # (b, n, n, h)
        x1_new, x2_new = self.node_conv(x1, x2, e)  # (b, n, h)
        x1 = x1 + self.node_norm1(x1_new)
        x2 = x2 + self.node_norm2(x2_new)
        return x1, x2, e
