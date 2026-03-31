import torch


class SinkhornDistance(torch.nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, max_iter, reduction="none"):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, mu, nu, C):
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = (
                self.eps
                * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1))
                + u
            )
            v = (
                self.eps
                * (
                    torch.log(nu + 1e-8)
                    - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)
                )
                + v
            )
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))
        self.actual_nits = actual_nits
        if self.reduction == "mean":
            cost = cost.mean()
        elif self.reduction == "sum":
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


def pairwise_distances(x, y):
    """
    Input: x is (B, N, D), y is (B, M, D)
    Output: dist is (B, N, M) with Euclidean distance squared
    """
    # x_norm shape: (B, N, 1)
    x_norm = (x**2).sum(2).view(x.shape[0], x.shape[1], 1)

    # y_norm shape: (B, 1, M) <--- FIXED: uses y.shape[1] (M), not y.shape[2] (D)
    y_norm = (y**2).sum(2).view(y.shape[0], 1, y.shape[1])

    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y.transpose(1, 2))
    return torch.clamp(dist, 0.0, float("inf"))


def get_sinkhorn_grads(x0_a, x0_b, sinkhorn_module, device):
    """
    Computes gradients of the Sinkhorn distance w.r.t inputs.
    Includes gradient normalization to ensure lambda behaves like a guidance scale.
    """
    with torch.enable_grad():
        # Detach to separate from previous graph
        var_a = x0_a.detach().requires_grad_(True)
        var_b = x0_b.detach().requires_grad_(True)

        # Reshape (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = var_a.shape
        flat_a = var_a.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        flat_b = var_b.permute(0, 2, 3, 1).contiguous().view(B, -1, C)

        # Compute Cost Matrix
        cost_matrix = pairwise_distances(flat_a, flat_b)

        # Define uniform measures
        n_points = flat_a.shape[1]
        mu = torch.empty(B, n_points, dtype=var_a.dtype, device=device).fill_(
            1.0 / n_points
        )
        nu = torch.empty(B, n_points, dtype=var_b.dtype, device=device).fill_(
            1.0 / n_points
        )

        # Forward pass
        loss, _, _ = sinkhorn_module(mu, nu, cost_matrix)

        # Compute Gradients
        grad_a, grad_b = torch.autograd.grad(loss.sum(), [var_a, var_b])

        # --- KEY FIX: Normalize Gradients ---
        # We normalize by the standard deviation to ensure the gradient
        # has a consistent magnitude regardless of the transport cost scale.
        grad_a_std = grad_a.std() + 1e-8
        grad_b_std = grad_b.std() + 1e-8

        grad_a = grad_a / grad_a_std
        grad_b = grad_b / grad_b_std

        return grad_a, grad_b
