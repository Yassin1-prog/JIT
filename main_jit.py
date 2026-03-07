import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr
import util.misc as misc

import copy
from engine_jit import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('JiT', add_help=False)

    # architecture
    parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=32, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--use_dual_ema', action='store_true',
                        help='Enable dual EMA tracking (both decay1 and decay2). Default: only decay1.')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=4, type=int) # per kaggle recommendation
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_images', default=500, type=int,
                        help='Number of images to generate')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')

    # dataset
    parser.add_argument('--data_path', default='./data/cifar10/test', type=str,
                        help='Path to the dataset root directory (should contain class folders directly)')
    parser.add_argument('--class_num', default=10, type=int)

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    # Distillation arguments
    parser.add_argument('--distill', action='store_true',
                        help='Enable knowledge distillation training')
    parser.add_argument('--teacher_model', default='JiT-S/2', type=str,
                        help='Teacher model architecture')
    parser.add_argument('--teacher_ckpt', default='', type=str,
                        help='Path to teacher checkpoint file')
    parser.add_argument('--alpha_vitkd', type=float, default=3e-5,
                        help='Weight for ViTKD shallow layer mimicking loss')
    parser.add_argument('--beta_vitkd', type=float, default=3e-6,
                        help='Weight for ViTKD deep layer generation loss')
    parser.add_argument('--lambda_vitkd', type=float, default=0.5,
                        help='Masking ratio for ViTKD generation loss')
    parser.add_argument('--mimic_only', action='store_true',
                        help='Only use shallow mimicking loss; skip the deep generation loss')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Set up TensorBoard logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # Data augmentation transforms
    transform_train = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor()
    ])

    dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train =", sampler_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create denoiser with optional distillation
    if args.distill:
        if not args.teacher_ckpt:
            raise ValueError("--teacher_ckpt must be provided when --distill is enabled")

        from distill_denoiser import DistillDenoiser
        model = DistillDenoiser(args)
        print(f"Using DistillDenoiser: student={args.model}, teacher={args.teacher_model}")
    else:
        from denoiser import Denoiser
        model = Denoiser(args)
        print(f"Using standard Denoiser: model={args.model}")

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # Resume from checkpoint if provided
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        # IN NEWER PYTORCH VERSIONS LIKE IN KAGGLE IT DEFAULTS TO TRUE AND BLOCKS LOADING CHECKPOINT.PTH
        # DUE TO SECUIRITY ISSUES. SO NEED TO SPECIFY weights_only=False, ALTERNATIVE SOLUTION IS :
        # Tell PyTorch that argparse.Namespace is safe to load
        #torch.serialization.add_safe_globals([argparse.Namespace])
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # A checkpoint saved from DistillDenoiser contains extra keys like
        # "teacher.*" and "vitkd_loss.*" that don't exist in a plain Denoiser.
        # Filter to only the keys the current model expects so evaluation without
        # --distill works on checkpoints produced by distillation training.
        checkpoint_model = checkpoint['model']
        model_keys = set(model_without_ddp.state_dict().keys())
        unexpected = [k for k in checkpoint_model if k not in model_keys]
        if unexpected:
            print(f"[INFO] Dropping {len(unexpected)} keys not in current model "
                  f"(e.g. {unexpected[0]}). This is expected when loading a "
                  f"DistillDenoiser checkpoint into a plain Denoiser.")
            checkpoint_model = {k: v for k, v in checkpoint_model.items() if k in model_keys}
        # strict=False: teacher.* keys are intentionally absent from checkpoints
        # (excluded by _drop_teacher in save_model) because teacher weights are
        # frozen and re-loaded from args.teacher_ckpt at DistillDenoiser.__init__.
        missing, unexpected2 = model_without_ddp.load_state_dict(checkpoint_model, strict=False)
        teacher_missing = [k for k in missing if k.startswith('teacher.')]
        non_teacher_missing = [k for k in missing if not k.startswith('teacher.')]
        if teacher_missing:
            print(f"[INFO] {len(teacher_missing)} teacher.* keys not in checkpoint "
                  f"(expected — teacher is re-loaded from teacher_ckpt at init).")
        if non_teacher_missing:
            raise RuntimeError(f"Unexpected missing keys in checkpoint: {non_teacher_missing}")

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        # teacher.* keys are intentionally omitted from checkpoints (frozen,
        # re-loaded on init); fall back to the current parameter value for them.
        model_without_ddp.ema_params1 = [
            ema_state_dict1[name].cuda() if name in ema_state_dict1 else param.data.clone()
            for name, param in model_without_ddp.named_parameters()
        ]
        model_without_ddp.ema_params2 = [
            ema_state_dict2[name].cuda() if name in ema_state_dict2 else param.data.clone()
            for name, param in model_without_ddp.named_parameters()
        ]
        print("Resumed checkpoint from", args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not args.evaluate_gen:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("Loaded optimizer & scaler state!")
        # If optimizer was saved from a DistillDenoiser checkpoint but will be loaded into a plain Denoiser (e.g. for evaluation),
        # the optimizer state dict contains extra keys like "vitkd_loss.*". But since optimizer has no effect in evaluation, just ignore loading it.
        elif 'epoch' in checkpoint:
            args.start_epoch = checkpoint['epoch'] + 1
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")

    # Evaluate generation
    if args.evaluate_gen:
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(model_without_ddp, args, 0, batch_size=args.gen_bsz, log_writer=log_writer)
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args)

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )

        if epoch % 100 == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer)
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
