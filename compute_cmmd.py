"""Compute CMMD between a directory of generated images and real reference images.

Either a pre-computed reference embeddings file or a real images directory must
be provided.

Usage (with cached reference embeddings — fast):
    python compute_cmmd.py \\
        --gen_dir output_dir/heun-steps50-cfg2.0-... \\
        --ref_embed_file fid_stats/jit_in32_cmmd-10000.npy \\
        --max_count 10000

Usage (computing reference embeddings on the fly — slower):
    python compute_cmmd.py \\
        --gen_dir output_dir/heun-steps50-cfg2.0-... \\
        --real_img_dir /path/to/real/images \\
        --max_count 10000
"""

import argparse
import os
import numpy as np

from cmmd.embedding import ClipEmbeddingModel
from cmmd.io_util import compute_embeddings_for_dir
from cmmd import distance as cmmd_distance


def get_args_parser():
    parser = argparse.ArgumentParser('Compute CMMD for generated images', add_help=False)
    parser.add_argument('--gen_dir', required=True, type=str,
                        help='Path to directory of generated images')
    parser.add_argument('--ref_embed_file', default='', type=str,
                        help='Path to pre-computed reference embeddings (.npy). '
                             'If provided, skips computing embeddings from --real_img_dir.')
    parser.add_argument('--real_img_dir', default='', type=str,
                        help='Path to real images directory. Used when --ref_embed_file '
                             'is not provided.')
    parser.add_argument('--max_count', default=10000, type=int,
                        help='Number of generated images to use. Should match the '
                             'reference set size for a symmetric comparison. '
                             'Defaults to 10000 (CIFAR-10 test set size).')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for CLIP inference')
    return parser


def main(args):
    if not args.ref_embed_file and not args.real_img_dir:
        raise ValueError('Provide either --ref_embed_file or --real_img_dir.')

    print('Loading CLIP embedding model...')
    clip_model = ClipEmbeddingModel()

    if args.ref_embed_file:
        print(f'Loading cached reference embeddings from: {args.ref_embed_file}')
        ref_embeddings = np.load(args.ref_embed_file)
    else:
        ref_embeddings = compute_embeddings_for_dir(
            args.real_img_dir, clip_model, batch_size=args.batch_size
        )

    eval_embeddings = compute_embeddings_for_dir(
        args.gen_dir, clip_model, batch_size=args.batch_size, max_count=args.max_count
    )

    cmmd_value = float(cmmd_distance.mmd(ref_embeddings, eval_embeddings).item())
    print(f'CMMD: {cmmd_value:.4f}')


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
