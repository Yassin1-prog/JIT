import os
import argparse
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from util.crop import center_crop_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to ImageNet root directory (should contain class folders directly)')
    parser.add_argument('--output_path', type=str, default='imagenet-train-256',
                        help='Folder where transformed images will be saved')
    parser.add_argument('--npz_dir', type=str, default='fid_stats',
                        help='Directory where the .npz file will be saved')
    parser.add_argument('--npz_name', type=str, default='imagenet_train.npz',
                        help='Name of the output .npz file')
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
    os.makedirs(args.npz_dir, exist_ok=True)

    to_pil = transforms.ToPILImage()
    global_idx = 0
    all_images = []

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

            # Collect image as numpy array (H, W, C) in uint8
            all_images.append(np.array(pil_img))

        print(f"Saved batch up to index={global_idx} ...")

    print("Finished saving all images.")

    # Stack all images and save as .npz
    print("Generating .npz file ...")
    images_np = np.stack(all_images, axis=0)  # Shape: (N, H, W, C)
    npz_path = os.path.join(args.npz_dir, args.npz_name)
    np.savez(npz_path, arr_0=images_np)
    print(f"Saved .npz file to: {npz_path}")
    print(f"Array shape: {images_np.shape}, dtype: {images_np.dtype}")


if __name__ == "__main__":
    main()