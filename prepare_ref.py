import os
import argparse
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from util.crop import center_crop_arr

# Code that correctly generates the valid .npz file to be run in a Kaggle Cell

"""
# Run this once in Kaggle to generate the correct stats file
import numpy as np
import torch
import torch_fidelity
from torch_fidelity.utils import (
    create_feature_extractor,
    extract_featuresdict_from_input_id_cached,
    resolve_feature_extractor,
    resolve_feature_layer_for_metric,
)
from torch_fidelity.metric_fid import fid_features_to_statistics

kwargs = dict(
    input1='/kaggle/input/datasets/ayush1220/cifar10/cifar10/test/',  # your reference images folder (flat is fine)
    input2=None,
    cuda=True,
    fid=True,
    verbose=True,
    batch_size=64,
    samples_find_deep=True,   # searches subdirs recursively
    samples_find_ext='png,jpg,jpeg',
    feature_extractor=None,
    feature_extractor_weights_path=None,
    feature_extractor_internal_dtype=None,
    feature_extractor_compile=False,
    cache=False,
    cache_root=None,
    input1_cache_name=None,
    rng_seed=2020,
    save_cpu_ram=False,
)

feature_extractor_name = resolve_feature_extractor(**kwargs)
feat_layer = resolve_feature_layer_for_metric('fid', **kwargs)
feat_extractor = create_feature_extractor(feature_extractor_name, [feat_layer], **kwargs)
featuresdict = extract_featuresdict_from_input_id_cached(1, feat_extractor, **kwargs)
stats = fid_features_to_statistics(featuresdict[feat_layer])

np.savez(
    '/kaggle/working/JIT/fid_stats/jit_in32_test_stats.npz',
    mu=stats['mu'],
    sigma=stats['sigma'],
)
print("Done. mu shape:", stats['mu'].shape, "sigma shape:", stats['sigma'].shape)
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to ImageNet root directory (should contain class folders directly)')
    parser.add_argument('--output_path', type=str, default='imagenet-train-256',
                        help='Folder where transformed images will be saved')
    parser.add_argument('--img_size', type=int, default=32,
                        help='Resolution to center-crop and resize')
    args = parser.parse_args()

    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
    ])

    dataset_train = datasets.ImageFolder(
        args.data_path,
        transform=transform_train
    )

    data_loader = DataLoader(
        dataset_train,
        batch_size=256,
        num_workers=32,
        shuffle=False,
        pin_memory=False
    )

    os.makedirs(args.output_path, exist_ok=True)

    to_pil = transforms.ToPILImage()
    global_idx = 0

    from tqdm import tqdm
    for batch_images, batch_labels in tqdm(data_loader):
        for i in range(batch_images.size(0)):
            img_tensor = batch_images[i]

            pil_img = to_pil(img_tensor)
            out_path = os.path.join(
                args.output_path,
                f"transformed_{global_idx:08d}.png"
            )
            pil_img.save(out_path, format='PNG', compress_level=0)
            global_idx += 1

        print(f"Saved batch up to index={global_idx} ...")

    print("Finished saving all images.")


if __name__ == "__main__":
    main()