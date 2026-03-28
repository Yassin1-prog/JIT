import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnKDLoss(nn.Module):
    """
    Attention Map KL-Divergence Knowledge Distillation Loss.

    Computes symmetric KL divergence between the head-averaged attention
    distributions of student and teacher across the last N transformer blocks.

    This module is parameter-free. All alignment (head-averaging, in-context
    stripping, renormalization) is performed upstream in JiT.forward before
    maps are passed here.

    Args:
        gamma_attn (float): Overall loss scale factor. Default: 1e-4.
        t_min (float): Timestep gate threshold. Samples with t <= t_min are
            excluded from the loss. At low t the input is dominated by Gaussian
            noise and attention maps are near-uniform, contributing only gradient
            noise. Default: 0.1.
        eps (float): Small constant added to attention weights before log to
            prevent log(0). Default: 1e-8.
        symmetric (bool): If True, compute 0.5*(KL(S||T) + KL(T||S)).
            If False, compute only KL(S||T) (student toward teacher).
            Default: True.
    """

    def __init__(self, gamma_attn: float = 1e-4, t_min: float = 0.1,
                 eps: float = 1e-8, symmetric: bool = True):
        super().__init__()
        self.gamma_attn = gamma_attn
        self.t_min = t_min
        self.eps = eps
        self.symmetric = symmetric

    def forward(
        self,
        attn_maps_s: list,
        attn_maps_t: list,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            attn_maps_s: List of ``num_attn_layers`` tensors, each ``[B, N, N]``.
                Head-averaged, in-context-stripped, renormalized student attention
                weights. Rows are distributions over N key positions.
            attn_maps_t: Same structure for the teacher (produced under
                ``torch.no_grad()``).
            t: ``[B]`` flattened timesteps in [0, 1], where 0 = pure noise
                and 1 = clean data.

        Returns:
            Scalar loss tensor (0.0 if all samples are below ``t_min``).
        """
        # ------------------------------------------------------------------ #
        # Timestep gate: exclude samples where t <= t_min.                   #
        # These samples have near-uniform attention due to high noise, which  #
        # would inject gradient noise rather than useful signal.             #
        # ------------------------------------------------------------------ #
        valid = t > self.t_min  # [B]
        if not valid.any():
            # Return a zero that is still part of the computation graph so
            # that .backward() doesn't fail if called unconditionally.
            return attn_maps_s[0].new_zeros(()).requires_grad_(attn_maps_s[0].requires_grad)

        total = attn_maps_s[0].new_zeros(())

        for a_s, a_t in zip(attn_maps_s, attn_maps_t):
            # a_s, a_t: [B, N, N]
            # Filter to valid timesteps only.
            a_s = a_s[valid]  # [B', N, N]
            a_t = a_t[valid]  # [B', N, N]
            B_, N, _ = a_s.shape

            # Add eps and renormalize to strict probability distributions.
            # The upstream renormalization after in-context stripping is
            # approximate; this final pass ensures correctness and handles
            # any residual numerical drift.
            a_s = a_s + self.eps
            a_s = a_s / a_s.sum(dim=-1, keepdim=True)

            a_t = a_t + self.eps
            a_t = a_t / a_t.sum(dim=-1, keepdim=True)

            # Reshape to [B'*N, N] so that F.kl_div with reduction='batchmean'
            # divides by B'*N, yielding average KL per query token per batch.
            a_s_flat = a_s.reshape(B_ * N, N)
            a_t_flat = a_t.reshape(B_ * N, N)

            # KL(S || T): F.kl_div(log_input, target) = sum(target * (log_target - log_input))
            # With input = log(a_s) and target = a_t this gives KL(T || S) in PyTorch's
            # convention, i.e. the student is pushed toward the teacher distribution.
            kl_s_t = F.kl_div(a_s_flat.log(), a_t_flat, reduction='batchmean')

            if self.symmetric:
                kl_t_s = F.kl_div(a_t_flat.log(), a_s_flat, reduction='batchmean')
                total = total + 0.5 * (kl_s_t + kl_t_s)
            else:
                total = total + kl_s_t

        # Average over layers so the loss magnitude is independent of num_attn_layers.
        return self.gamma_attn * (total / len(attn_maps_s))
