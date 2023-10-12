import torch
from torch import nn
from torch.nn import functional as F


class ResnetPointnet(nn.Module):
    """PointNet-based encoder network with ResNet blocks.

    Args:
        out_dim (int): dimension of latent code c
        hidden_dim (int): hidden dimension of the network
        dim (int): input dimensionality (default 3)
    """

    def __init__(self, out_dim, hidden_dim, dim=3, **kwargs):
        super().__init__()
        self.out_dim = out_dim

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.use_block2 = kwargs.get("use_block2", False)
        self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim) if self.use_block2 else None
        self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, out_dim)

        self.act = nn.ReLU()

    @staticmethod
    def pool(x, dim=-1, keepdim=False):
        return x.max(dim=dim, keepdim=keepdim)[0]

    def forward(self, p):
        B, N, feat_dim = p.shape
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        c = self.fc_c(self.act(net))
        c = F.softmax(c.view(-1, self.out_dim), dim=-1)
        c = c.view(B, N, self.out_dim)

        return c


class ResnetBlockFC(nn.Module):
    """Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        self.shortcut = nn.Linear(size_in, size_out, bias=False)

        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))
        x_s = self.shortcut(x)

        return x_s + dx


class get_part_seg_loss(torch.nn.Module):
    def forward(self, pred, target, trans_feat):
        return F.nll_loss(torch.log(pred), target)
