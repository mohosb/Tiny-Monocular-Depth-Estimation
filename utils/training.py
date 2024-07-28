import torch
import torch.nn.functional as F
import numpy as np

def identity(arg):
    return arg

class ReducableLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        if reduction == 'mean':
            self.reduce = torch.mean
        elif reduction == 'sum':
            self.reduce = torch.sum
        elif reduction == 'none':
            self.reduce = identity
        else:
            raise Exception('Argument "reduction" must be "mean", "sum" or "none"!')


class ScaleInvariantMSELoss(ReducableLoss):
    def __init__(self, use_log=True, reduction='mean', eps=1e-8):
        super().__init__(reduction)
        self.use_log = use_log
        self.eps = eps

    def forward(self, z, y):
        if self.use_log:
            g = (z + self.eps).log() - (y + self.eps).log()
        else:
            g = z - y
        dim = torch.arange(1, g.dim()).tolist()
        n = np.prod(g.shape[1:])
        return self.reduce(
            (g ** 2).sum(dim) / n - (g.sum(dim) ** 2) / (n ** 2)
        )

class ScaleInvariantRMSELoss(ReducableLoss):
    def __init__(self, use_log=True, reduction='mean', eps=1e-8):
        super().__init__(reduction)
        self.use_log = use_log
        self.eps = eps

    def forward(self, z, y):
        if self.use_log:
            g = (z + self.eps).log() - (y + self.eps).log()
        else:
            g = z - y
        dim = torch.arange(1, g.dim()).tolist()
        n = np.prod(g.shape[1:])
        return self.reduce(
            ((g ** 2).sum(dim) / n - (g.sum(dim) ** 2) / (n ** 2)).sqrt()
        )

class GradientMatchingLoss(ReducableLoss):
    def __init__(self, scales=(1.0, 0.5, 0.25, 0.125), use_log=True, reduction='mean', eps=1e-8):
        super().__init__(reduction)
        self.scales = scales
        self.use_log = use_log
        self.eps = eps
        self.grad_conv_x = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=False)
        self.grad_conv_y = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=False)
        self.grad_conv_x.weight = torch.nn.Parameter(
            torch.tensor([[[[ 1.,  0., -1.],
                            [ 2.,  0., -2.],
                            [ 1.,  0., -1.]]]]),
            requires_grad=False
        )
        self.grad_conv_y.weight = torch.nn.Parameter(
            torch.tensor([[[[ 1.,  2.,  1.],
                            [ 0.,  0.,  0.],
                            [-1., -2., -1.]]]]),
            requires_grad=False
        )

    def forward(self, z, y):
        n = np.prod(y.shape[1:])

        cummulator = []
        for s in self.scales:
            scaled_z = F.interpolate(z, scale_factor=s, mode='bilinear', align_corners=False, antialias=False)
            scaled_y = F.interpolate(y, scale_factor=s, mode='bilinear', align_corners=False, antialias=False)

            if self.use_log:
                g = (scaled_z + self.eps).log() - (scaled_y + self.eps).log()
            else:
                g = scaled_z - scaled_y
            dim = torch.arange(1, g.dim()).tolist()

            grad_x = self.grad_conv_x(scaled_z)
            grad_y = self.grad_conv_x(scaled_z)

            cummulator.append(
                ((grad_x * g).abs() + (grad_y * g).abs()).sum(dim)
            )

        return self.reduce(sum(cummulator) / n)

class AbsoluteRelativeLoss(ReducableLoss):
    def __init__(self, reduction='mean', eps=1e-8):
        super().__init__(reduction)
        self.eps = eps

    def forward(self, z, y):
        n = np.prod(y.shape[1:])
        dim = torch.arange(1, y.dim()).tolist()
        return self.reduce(
            ((y - z).abs() / (y + self.eps)).sum(dim) / n
        )

class PairwiseDistillationLoss(ReducableLoss):
    def __init__(self, reduction='mean'):
        super().__init__(reduction)

    def forward(self, z, y):
        flat_z = z.flatten(2)
        flat_y = z.flatten(2)

        dif_z = flat_z.unsqueeze(3) - flat_z.unsqueeze(2)
        dif_y = flat_y.unsqueeze(3) - flat_y.unsqueeze(2)

        dim = torch.arange(1, y.dim()).tolist()
        return self.reduce(
            (dif_z - dif_y).abs().mean(dim)
        )

class WeightedComposedLoss(torch.nn.Module):
    def __init__(self, weights_and_losses):
        super().__init__()
        self.losses = torch.nn.ModuleList(l for _, l in weights_and_losses)
        self.weights = [w for w, _ in weights_and_losses]

    def forward(self, z, y):
        return sum(w * l(z, y) for w, l in zip(self.weights, self.losses))

@torch.no_grad()
def evaluate(teacher_model, student_model, criterion, test_loader):
    losses = []
    teacher_model.eval()
    student_model.eval()
    for x in test_loader:
        x = list(x.numpy())

        y = teacher_model(x)
        z = student_model(x)

        loss = criterion(z, y)
        losses.append(loss.cpu())
    return torch.cat(tuple(losses)).mean().item()

