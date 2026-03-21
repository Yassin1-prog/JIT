"""Pre-compute and cache CLIP reference embeddings for CMMD evaluation.

Usage:
    python precompute_cmmd_stats.py --img_dir /path/to/real_images \
                                    --output cmmd_stats/jit_in32_cmmd.npy \
                                    --batch_size 32 \
                                    --max_count 10000
"""

import argparse
import os
import numpy as np

from cmmd.embedding import ClipEmbeddingModel
from cmmd.io_util import compute_embeddings_for_dir


def get_args_parser():
    parser = argparse.ArgumentParser('Precompute CMMD reference embeddings', add_help=False)
    parser.add_argument('--img_dir', required=True, type=str,
                        help='Path to directory of reference images')
    parser.add_argument('--output', required=True, type=str,
                        help='Output .npy file path for cached embeddings')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for CLIP inference')
    parser.add_argument('--max_count', default=10000, type=int,
                        help='Maximum number of images to use (-1 for all). '
                             'Defaults to 10000 to match CIFAR-10 test set size.')
    return parser


def main(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"Loading CLIP embedding model...")
    model = ClipEmbeddingModel()

    print(f"Computing embeddings for images in: {args.img_dir}")
    embeddings = compute_embeddings_for_dir(
        args.img_dir, model, batch_size=args.batch_size, max_count=args.max_count
    )

    np.save(args.output, embeddings)
    print(f"Saved {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]}) to: {args.output}")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
