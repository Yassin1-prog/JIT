# --------------------------------------------------------
# Weight Selection Initialization for JIT Models
# Based on: "Initializing Models with Larger Ones" (ICLR 2024)
# --------------------------------------------------------
"""Initialize a smaller JiT model from a larger teacher checkpoint.

This script performs weight selection (uniform per-dimension subsampling when
needed) to transfer compatible parameters from a larger teacher model to a
smaller student model.

Example:
    python weight_selection_jit.py ^
      --teacher_ckpt .\checkpoints\jit_s2_ckpt.pt ^
      --teacher_model JiT-S/2 ^
      --student_model JiT-T/2 ^
      --output .\checkpoints\jit_t2_init_from_s2.pt ^
      --img_size 32 ^
      --class_num 10

Output:
    Saves a resume-compatible PyTorch checkpoint at ``--output`` with keys:
    - ``model``: full initialized student state dict
    - ``model_ema1``: same as ``model``
    - ``model_ema2``: same as ``model``
    - ``epoch``: -1 (so training resume logic starts at epoch 0)
"""

import argparse
import torch

from denoiser import Denoiser


# Static config extracted from model_jit.py factory functions.
# Used for lightweight compatibility checks without model instantiation.
JIT_CONFIGS = {
    'JiT-T/2':  dict(patch_size=2,  hidden=384,  depth=6,  heads=6,  bottleneck=0,   in_ctx_len=8),
    'JiT-S/2':  dict(patch_size=2,  hidden=512,  depth=10, heads=8,  bottleneck=0,   in_ctx_len=8),
    'JiT-B/16': dict(patch_size=16, hidden=768,  depth=12, heads=12, bottleneck=128,  in_ctx_len=32),
    'JiT-B/32': dict(patch_size=32, hidden=768,  depth=12, heads=12, bottleneck=128,  in_ctx_len=32),
    'JiT-L/16': dict(patch_size=16, hidden=1024, depth=24, heads=16, bottleneck=128,  in_ctx_len=32),
    'JiT-L/32': dict(patch_size=32, hidden=1024, depth=24, heads=16, bottleneck=128,  in_ctx_len=32),
    'JiT-H/16': dict(patch_size=16, hidden=1280, depth=32, heads=16, bottleneck=256,  in_ctx_len=32),
    'JiT-H/32': dict(patch_size=32, hidden=1280, depth=32, heads=16, bottleneck=256,  in_ctx_len=32),
}

# Keys ending with these suffixes are skipped (non-learnable or regenerated).
SKIP_SUFFIXES = ('pos_embed', 'freqs_cos', 'freqs_sin')


def uniform_element_selection(wt, s_shape):
    """Select a subset of teacher weights by uniform sampling per dimension.

    For each dimension, if teacher_dim is divisible by student_dim, use a
    regular step; otherwise use evenly-spaced indices via linspace + rounding.
    """
    assert wt.dim() == len(s_shape), \
        f"Tensor rank mismatch: teacher has {wt.dim()} dims, student shape has {len(s_shape)}"
    ws = wt.clone()
    for dim in range(wt.dim()):
        assert wt.shape[dim] >= s_shape[dim], \
            f"Teacher dim {dim} ({wt.shape[dim]}) < student ({s_shape[dim]})"
        if wt.shape[dim] == s_shape[dim]:
            continue
        if wt.shape[dim] % s_shape[dim] == 0:
            step = wt.shape[dim] // s_shape[dim]
            indices = torch.arange(s_shape[dim]) * step
        else:
            indices = torch.round(
                torch.linspace(0, wt.shape[dim] - 1, s_shape[dim])
            ).long()
        ws = torch.index_select(ws, dim, indices)
    return ws


def validate_compatibility(teacher_name, student_name):
    """Check that teacher→student transfer is valid. Returns (teacher_cfg, student_cfg).

    Raises ValueError if either model is unknown or teacher is not strictly
    larger than student. Prints a warning for cross-patch-size transfers.
    """
    if teacher_name not in JIT_CONFIGS:
        raise ValueError(f"Unknown teacher model: '{teacher_name}'. "
                         f"Valid names: {list(JIT_CONFIGS.keys())}")
    if student_name not in JIT_CONFIGS:
        raise ValueError(f"Unknown student model: '{student_name}'. "
                         f"Valid names: {list(JIT_CONFIGS.keys())}")

    t_cfg = JIT_CONFIGS[teacher_name]
    s_cfg = JIT_CONFIGS[student_name]

    if t_cfg['hidden'] < s_cfg['hidden']:
        raise ValueError(
            f"Teacher hidden_size ({t_cfg['hidden']}) < student ({s_cfg['hidden']}). "
            f"Teacher must be larger than student.")
    if t_cfg['depth'] < s_cfg['depth']:
        raise ValueError(
            f"Teacher depth ({t_cfg['depth']}) < student ({s_cfg['depth']}). "
            f"Teacher must be deeper than student.")

    if t_cfg['patch_size'] != s_cfg['patch_size']:
        print(f"[weight-init] WARNING: Cross-patch-size transfer "
              f"({teacher_name} ps={t_cfg['patch_size']} → "
              f"{student_name} ps={s_cfg['patch_size']}). "
              f"x_embedder and final_layer.linear will be initialized from scratch. "
              f"All other weights (~97-99% of params) transfer normally.")

    return t_cfg, s_cfg


def load_teacher_state_dict(ckpt_path, use_ema=True):
    """Load teacher weights from a JIT checkpoint.

    Prefers EMA weights (model_ema1) by default since they are higher quality
    and used for evaluation in the JIT pipeline.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if use_ema and 'model_ema1' in ckpt:
        print("[weight-init] Using EMA weights from teacher checkpoint.")
        return ckpt['model_ema1']
    if 'model' in ckpt:
        if use_ema:
            print("[weight-init] EMA weights not found, falling back to model weights.")
        return ckpt['model']
    # Raw state dict fallback (e.g. pre-computed weight file)
    return ckpt


def apply_weight_selection(teacher_sd, student_model, cross_patch_size=False, verbose=True):
    """Apply weight selection from teacher state dict to student model.

    Returns a state dict compatible with student_model.load_state_dict(strict=False).
    """
    student_sd = student_model.state_dict()
    selected = {}
    n_copied, n_selected, n_skipped = 0, 0, 0
    skipped_keys = []

    for key, student_tensor in student_sd.items():
        # 1. Skip non-learnable / regenerated parameters
        if any(key.endswith(s) for s in SKIP_SUFFIXES):
            n_skipped += 1
            skipped_keys.append(key)
            continue

        # 2. Skip final_layer.linear in cross-patch-size mode
        if cross_patch_size and 'final_layer.linear' in key:
            n_skipped += 1
            skipped_keys.append(key)
            if verbose:
                print(f"  [skip] {key} (cross-patch-size: output layout mismatch)")
            continue

        # 3. Skip if key not in teacher
        if key not in teacher_sd:
            n_skipped += 1
            skipped_keys.append(key)
            if verbose:
                print(f"  [skip] {key} (not found in teacher)")
            continue

        teacher_tensor = teacher_sd[key]

        # 4. Skip if tensor rank differs (safety guard)
        if teacher_tensor.dim() != student_tensor.dim():
            n_skipped += 1
            skipped_keys.append(key)
            if verbose:
                print(f"  [skip] {key} (rank mismatch: teacher {teacher_tensor.dim()}D "
                      f"vs student {student_tensor.dim()}D)")
            continue

        # 5. Skip if teacher is smaller than student in any dimension
        if any(t < s for t, s in zip(teacher_tensor.shape, student_tensor.shape)):
            n_skipped += 1
            skipped_keys.append(key)
            if verbose:
                print(f"  [skip] {key} (teacher shape {list(teacher_tensor.shape)} "
                      f"< student {list(student_tensor.shape)} in some dim)")
            continue

        # 6. Direct copy if shapes match
        if teacher_tensor.shape == student_tensor.shape:
            selected[key] = teacher_tensor.clone()
            n_copied += 1
            continue

        # 7. Uniform element selection
        selected[key] = uniform_element_selection(teacher_tensor, student_tensor.shape)
        n_selected += 1
        if verbose:
            print(f"  [select] {key}: {list(teacher_tensor.shape)} → {list(student_tensor.shape)}")

    print(f"[weight-init] Summary: {n_copied} copied, {n_selected} selected (uniform), "
          f"{n_skipped} skipped")
    if verbose and skipped_keys:
        print(f"[weight-init] Skipped keys: {skipped_keys}")

    return selected


def _make_student_args(student_model, img_size, class_num):
    """Create a minimal args namespace for Denoiser instantiation."""
    return argparse.Namespace(
        model=student_model,
        img_size=img_size,
        class_num=class_num,
        attn_dropout=0.0,
        proj_dropout=0.0,
        label_drop_prob=0.1,
        P_mean=-0.8,
        P_std=0.8,
        noise_scale=1.0,
        t_eps=5e-2,
        ema_decay1=0.9999,
        ema_decay2=0.9996,
        use_dual_ema=False,
        sampling_method='euler',
        num_sampling_steps=50,
        cfg=1.0,
        interval_min=0.0,
        interval_max=1.0,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Weight Selection: initialize a smaller JiT model from a larger one')
    parser.add_argument('--teacher_ckpt', type=str, required=True,
                        help='Path to teacher JiT checkpoint')
    parser.add_argument('--teacher_model', type=str, required=True,
                        help='Teacher model name (e.g. JiT-S/2)')
    parser.add_argument('--student_model', type=str, required=True,
                        help='Student model name (e.g. JiT-T/2)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for the resume-compatible checkpoint')
    parser.add_argument('--img_size', type=int, default=32,
                        help='Image size for the student model (default: 32)')
    parser.add_argument('--class_num', type=int, default=10,
                        help='Number of classes (default: 10)')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Use EMA weights from teacher (default: True)')
    parser.add_argument('--no_ema', dest='use_ema', action='store_false',
                        help='Use raw model weights instead of EMA')
    args = parser.parse_args()

    # Validate compatibility
    t_cfg, s_cfg = validate_compatibility(args.teacher_model, args.student_model)
    cross_patch = (t_cfg['patch_size'] != s_cfg['patch_size'])

    # Instantiate student (lightweight — only for state dict structure)
    student_args = _make_student_args(args.student_model, args.img_size, args.class_num)
    student = Denoiser(student_args)

    # Load teacher and apply weight selection
    teacher_sd = load_teacher_state_dict(args.teacher_ckpt, use_ema=args.use_ema)
    selected = apply_weight_selection(teacher_sd, student, cross_patch_size=cross_patch)

    # Load selected weights into student to get a complete state dict
    # (selected keys get teacher values, missing keys keep initialize_weights() values)
    student.load_state_dict(selected, strict=False)
    full_sd = student.state_dict()

    # Save as resume-compatible checkpoint
    checkpoint = {
        'model': full_sd,
        'model_ema1': full_sd,
        'model_ema2': full_sd,
        'epoch': -1,  # so start_epoch = -1 + 1 = 0
    }
    torch.save(checkpoint, args.output)
    print(f"[weight-init] Saved resume-compatible checkpoint to {args.output}")


if __name__ == '__main__':
    main()
