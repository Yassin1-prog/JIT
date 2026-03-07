import math
import sys
import os
import shutil
import time

import torch
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
import copy



def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # normalize image to [-1, 1]
        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # Log distillation losses if using DistillDenoiser
        if hasattr(model_without_ddp, 'loss_vitkd'):
            metric_logger.update(loss_x=model_without_ddp.loss_x.item())
            metric_logger.update(loss_vitkd=model_without_ddp.loss_vitkd.item())

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)

                # Log distillation losses to TensorBoard
                if hasattr(model_without_ddp, 'loss_vitkd'):
                    log_writer.add_scalar('loss_x', model_without_ddp.loss_x.item(), epoch_1000x)
                    log_writer.add_scalar('loss_vitkd', model_without_ddp.loss_vitkd.item(), epoch_1000x)


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = args.num_images // (batch_size * world_size) + 1

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    # ensure that the number of images per class is equal.
    class_num = args.class_num
    assert args.num_images % class_num == 0, "Number of images per class must be the same"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    # Timing metrics
    total_inference_time = 0.0
    total_images_generated = 0

    # Memory tracking
    total_peak_memory_mb = 0.0

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        # CUDA sync to ensure all previous operations complete
        torch.cuda.synchronize()

        # Reset peak memory stats to isolate this batch's memory usage
        torch.cuda.reset_peak_memory_stats()

        # Start timing
        batch_start_time = time.time()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(labels_gen)

        # CUDA sync to ensure generation completes before measuring
        torch.cuda.synchronize()

        # End timing
        batch_end_time = time.time()
        batch_inference_time = batch_end_time - batch_start_time

        # Accumulate timing
        total_inference_time += batch_inference_time

        # Measure peak memory for this batch
        MB = 1024.0 * 1024.0
        batch_peak_memory_mb = torch.cuda.max_memory_allocated() / MB
        total_peak_memory_mb += batch_peak_memory_mb

        if misc.is_dist_avail_and_initialized():
            torch.distributed.barrier()

        # denormalize images
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        # distributed save images
        # Track how many images from this batch were actually used
        batch_images_count = 0
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            batch_images_count += 1
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

        total_images_generated += batch_images_count

    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()

    # Aggregate timing metrics across all distributed processes
    if misc.is_dist_avail_and_initialized():
        # Convert to tensors for all_reduce
        timing_tensor = torch.tensor([total_inference_time, float(total_images_generated)],
                                      dtype=torch.float64, device='cuda')
        torch.distributed.all_reduce(timing_tensor, op=torch.distributed.ReduceOp.SUM)

        # Extract aggregated values
        total_inference_time_all_ranks = timing_tensor[0].item()
        total_images_generated_all_ranks = int(timing_tensor[1].item())
    else:
        total_inference_time_all_ranks = total_inference_time
        total_images_generated_all_ranks = total_images_generated

    # Compute derived metrics
    avg_time_per_image_ms = (total_inference_time_all_ranks / total_images_generated_all_ranks) * 1000
    throughput_images_per_sec = total_images_generated_all_ranks / total_inference_time_all_ranks

    print(f"\n=== Inference Timing Metrics ===")
    print(f"Total inference time: {total_inference_time_all_ranks:.2f} seconds")
    print(f"Total images generated: {total_images_generated_all_ranks}")
    print(f"Average time per image: {avg_time_per_image_ms:.2f} ms/image")
    print(f"Throughput: {throughput_images_per_sec:.2f} images/second")
    print(f"================================\n")

    # Compute average peak memory
    avg_peak_memory_mb = total_peak_memory_mb / num_steps
    avg_peak_memory_gb = avg_peak_memory_mb / 1024.0

    print(f"Average peak GPU memory: {avg_peak_memory_gb:.3f} GB ({avg_peak_memory_mb:.1f} MB)")

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)

    # compute FID and IS
    if log_writer is not None:
        if args.img_size == 256:
            fid_statistics_file = 'fid_stats/jit_in256_stats.npz'
        elif args.img_size == 512:
            fid_statistics_file = 'fid_stats/jit_in512_stats.npz'
        elif args.img_size == 32:
            # for debugging will be initially for the 10k split of cifar10
            fid_statistics_file = 'fid_stats/jit_in32_test_stats.npz'
        else:
            raise NotImplementedError
        real_img_dir = args.real_img_dir if args.real_img_dir else None
        compute_kid = real_img_dir is not None
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=real_img_dir,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=compute_kid,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))

        if compute_kid:
            kid = metrics_dict['kernel_inception_distance_mean']
            log_writer.add_scalar('kid{}'.format(postfix), kid, epoch)
            print("KID: {:.4f}".format(kid))
        
        # Log inference timing metrics
        log_writer.add_scalar('inference_time_total_sec{}'.format(postfix),
                              total_inference_time_all_ranks, epoch)
        log_writer.add_scalar('inference_time_per_image_ms{}'.format(postfix),
                              avg_time_per_image_ms, epoch)
        log_writer.add_scalar('inference_throughput_imgs_per_sec{}'.format(postfix),
                              throughput_images_per_sec, epoch)

        # Log GPU memory metric
        log_writer.add_scalar('gpu_memory_avg_peak_gb{}'.format(postfix),
                              avg_peak_memory_gb, epoch)

        shutil.rmtree(save_folder)

    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()
