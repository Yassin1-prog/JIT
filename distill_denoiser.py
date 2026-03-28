import torch
import torch.nn as nn
from model_jit import JiT_models
from VITKD.vitkd import ViTKDLoss


class DistillDenoiser(nn.Module):
    """
    Knowledge Distillation wrapper for JiT models using ViTKD.

    Combines flow-matching denoising loss with feature-based distillation.
    The teacher model is frozen and only used for feature extraction.
    """

    def __init__(self, args):
        super().__init__()

        # ====================================================================
        # STUDENT MODEL (Trainable)
        # ====================================================================
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        # ====================================================================
        # TEACHER MODEL (Frozen)
        # ====================================================================
        self.teacher = JiT_models[args.teacher_model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=0.0,  # No dropout for teacher
            proj_drop=0.0,
        )

        # Load teacher checkpoint
        if args.teacher_ckpt:
            checkpoint = torch.load(args.teacher_ckpt, map_location='cpu', weights_only=False)
            # Extract model state dict (handle both direct state_dict and checkpoint dict)
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_ema1' in checkpoint:
                # Use EMA weights for teacher (typically better)
                state_dict = checkpoint['model_ema1']
            else:
                state_dict = checkpoint

            # The checkpoint may have been saved from a Denoiser/DistillDenoiser
            # wrapper where JiT lives under self.net, so keys carry a "net." prefix.
            # Strip the prefix so the bare JiT model can load them.
            if any(k.startswith('net.') for k in state_dict.keys()):
                state_dict = {k[len('net.'):]: v for k, v in state_dict.items() if k.startswith('net.')}

            self.teacher.load_state_dict(state_dict)
            print(f"Loaded teacher checkpoint from {args.teacher_ckpt}")

        # Freeze teacher completely
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # ====================================================================
        # VITKD LOSS MODULE
        # ====================================================================
        # Get dimensions from model configs
        student_dims = self.net.hidden_size
        teacher_dims = self.teacher.hidden_size

        # Instantiate ViTKD loss without mmcls registry
        self.vitkd_loss = ViTKDLoss(
            name='loss_vitkd',
            use_this=True,
            student_dims=student_dims,
            teacher_dims=teacher_dims,
            alpha_vitkd=args.alpha_vitkd,
            beta_vitkd=args.beta_vitkd,
            lambda_vitkd=args.lambda_vitkd,
            mimic_only=getattr(args, 'mimic_only', False),
        )

        # ====================================================================
        # DENOISING PARAMETERS (same as Denoiser)
        # ====================================================================
        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # EMA parameters (for student only)
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # Pre-compute indices of non-teacher parameters so update_ema() can
        # skip the frozen teacher without a per-step string comparison.
        self._ema_trainable_indices = [
            i for i, (name, _) in enumerate(self.named_parameters())
            if not name.startswith('teacher.')
        ]

        # Generation parameters
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels):
        """
        Forward pass with distillation.

        Returns:
            loss: Scalar loss combining denoising + distillation

        When self.training=True:
            loss = loss_x + loss_vitkd
        When self.training=False (evaluation):
            loss = loss_x only
        """
        labels_dropped = self.drop_labels(labels) if self.training else labels

        # ====================================================================
        # FLOW MATCHING SETUP (same as Denoiser)
        # ====================================================================
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        # ====================================================================
        # STUDENT FORWARD PASS (with features)
        # ====================================================================
        x_pred, feats_s = self.net(z, t.flatten(), labels_dropped, return_features=True)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # Flow-matching loss
        loss_x = ((v - v_pred) ** 2).mean(dim=(1, 2, 3)).mean()

        # ====================================================================
        # TEACHER FORWARD PASS (frozen, no gradients)
        # ====================================================================
        if self.training:
            with torch.no_grad():
                # Teacher must be in eval mode
                self.teacher.eval()

                # Get teacher features (no need for predictions)
                _, feats_t = self.teacher(z, t.flatten(), labels_dropped, return_features=True)

            # Compute ViTKD loss
            # feats_s = [low_s, high_s] where low_s: [B, 2, N, D_s], high_s: [B, N, D_s]
            # feats_t = [low_t, high_t] where low_t: [B, 2, N, D_t], high_t: [B, N, D_t]
            loss_vitkd = self.vitkd_loss(feats_s, feats_t)

            # Total loss
            loss = loss_x + loss_vitkd

            # Store individual losses for logging (attach as attributes)
            self.loss_x = loss_x.detach()
            self.loss_vitkd = loss_vitkd.detach()
        else:
            # During evaluation, only use denoising loss
            loss = loss_x
            self.loss_x = loss_x.detach()
            self.loss_vitkd = torch.tensor(0.0, device=loss.device)

        return loss

    # ====================================================================
    # GENERATION METHODS (copied from Denoiser)
    # ====================================================================
    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        # Use student network for generation
        x_cond = self.net(z, t.flatten(), labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        """Update EMA for student + vitkd_loss parameters only (skip frozen teacher)."""
        params = list(self.parameters())
        for i in self._ema_trainable_indices:
            param = params[i]
            self.ema_params1[i].detach().mul_(self.ema_decay1).add_(param.detach(), alpha=1 - self.ema_decay1)
            self.ema_params2[i].detach().mul_(self.ema_decay2).add_(param.detach(), alpha=1 - self.ema_decay2)
