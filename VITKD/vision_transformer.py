# Copyright (c) OpenMMLab. All rights reserved.
# =============================================================================
# This file implements the Vision Transformer (ViT) backbone, which is the
# core model from the paper "An Image is Worth 16x16 Words". The idea is:
#   1. Split an image into small patches (e.g., 16x16 pixels each).
#   2. Treat each patch as a "token" (like a word in a sentence).
#   3. Feed these tokens through a stack of Transformer encoder layers.
#   4. Use the output for image classification.
#
# This version has been MODIFIED for Knowledge Distillation (KD):
#   - The forward() method extracts features from early, middle, and late
#     layers so a student model can learn from a teacher model's internal
#     representations.
# =============================================================================

from typing import Sequence  # For type-hinting lists/tuples of indices

import numpy as np            # Numerical operations (e.g., linspace for drop rates)
import torch                  # The core PyTorch library
import torch.nn as nn         # Neural network building blocks (Linear, Dropout, etc.)
from mmcv.cnn import build_norm_layer                   # Helper to build normalization layers (e.g., LayerNorm)
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed # FFN = Feed-Forward Network; PatchEmbed = splits image into patch embeddings
from mmengine.model import BaseModule, ModuleList        # Base classes for models and lists of modules
from mmengine.model.weight_init import trunc_normal_     # Truncated normal distribution for weight initialization

from mmcls.registry import MODELS                               # Registry to register this model so configs can find it
from ..utils import MultiheadAttention, resize_pos_embed, to_2tuple  # Attention module, position embedding resizer, tuple helper
from .base_backbone import BaseBackbone                          # Base class all backbones inherit from


# =============================================================================
# TransformerEncoderLayer: ONE single layer of the Transformer encoder.
#
# Each layer does two things:
#   1. Multi-Head Self-Attention (MHSA): lets each token look at ALL other
#      tokens to understand relationships (e.g., "this patch is near that patch").
#   2. Feed-Forward Network (FFN): a small neural network applied to each
#      token independently to transform its features.
#
# Both steps use "residual connections" (adding the input back to the output)
# and "layer normalization" (normalizing values to stabilize training).
#
# The pattern is:  x -> Norm -> Attention -> Add residual
#                  x -> Norm -> FFN       -> Add residual
# =============================================================================
class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,          # Size of each token's feature vector (e.g., 768 for ViT-Base)
                 num_heads,           # Number of attention heads (each head attends independently)
                 feedforward_channels,# Hidden size inside the FFN (usually 4x embed_dims)
                 drop_rate=0.,        # Dropout probability for FFN outputs
                 attn_drop_rate=0.,   # Dropout probability for attention weights
                 drop_path_rate=0.,   # DropPath probability (stochastic depth — randomly skip layers during training)
                 num_fcs=2,           # Number of fully-connected layers in the FFN
                 qkv_bias=True,       # Whether to add a bias term to Query/Key/Value projections
                 act_cfg=dict(type='GELU'),  # Activation function config (GELU is smooth ReLU)
                 norm_cfg=dict(type='LN'),   # Normalization config (LN = LayerNorm)
                 init_cfg=None):
        super(TransformerEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims

        # --- Layer Normalization #1 (applied BEFORE attention) ---
        # LayerNorm normalizes each token's features to have zero mean and unit variance.
        # This helps training converge faster and more stably.
        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        # --- Multi-Head Self-Attention ---
        # Each token creates a Query (Q), Key (K), and Value (V) vector.
        # Attention score = softmax(Q @ K^T / sqrt(d)).  Output = score @ V.
        # "Multi-head" means we do this multiple times in parallel with different
        # learned projections, then concatenate the results.
        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias)

        # --- Layer Normalization #2 (applied BEFORE the FFN) ---
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        # --- Feed-Forward Network (FFN) ---
        # A 2-layer MLP: Linear(embed_dims -> feedforward_channels) -> GELU -> Linear(feedforward_channels -> embed_dims)
        # This gives each token the capacity to transform its features non-linearly.
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    @property
    def norm1(self):
        """Convenience property to access the first LayerNorm by name."""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """Convenience property to access the second LayerNorm by name."""
        return getattr(self, self.norm2_name)

    def init_weights(self):
        """Initialize weights for the FFN's linear layers.
        Xavier uniform helps keep gradients well-scaled at the start of training.
        """
        super(TransformerEncoderLayer, self).init_weights()
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        """Forward pass of one Transformer encoder layer.

        Args:
            x: Tensor of shape (batch_size, num_tokens, embed_dims)

        The two core operations with residual connections:
          1. x = x + Attention(LayerNorm(x))   -- self-attention with skip connection
          2. x = x + FFN(LayerNorm(x))         -- feed-forward with skip connection
        """
        # Step 1: Normalize -> Attention -> Add back the original x (residual)
        x = x + self.attn(self.norm1(x))
        # Step 2: Normalize -> FFN -> Add back the original x (residual)
        # The `identity=x` parameter tells FFN to use x as the residual
        x = self.ffn(self.norm2(x), identity=x)
        return x


# =============================================================================
# VisionTransformer: The full ViT backbone model.
#
# HIGH-LEVEL FLOW:
#   Input Image (e.g., 224x224x3)
#     |
#     v
#   [Patch Embedding] -- splits image into patches, projects each to a vector
#     |                  e.g., 224/16 = 14 patches per side -> 196 patches total
#     |                  each patch becomes a 768-dim vector (for ViT-Base)
#     v
#   [Prepend CLS Token] -- a special learnable token prepended to the sequence
#     |                     used later for classification (like a "summary" token)
#     v
#   [Add Position Embeddings] -- adds learnable vectors so the model knows
#     |                          WHERE each patch came from in the image
#     v
#   [Transformer Encoder x N] -- N stacked encoder layers (e.g., 12 for Base)
#     |
#     v
#   [Output] -- for KD: returns low-level features, high-level features,
#               mid-layer CLS token, and final CLS token
# =============================================================================
@MODELS.register_module()  # Register so the config system can instantiate this class by name
class VisionTransformer(BaseBackbone):
    """Vision Transformer.

    A PyTorch implement of : `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        arch (str | dict): Vision Transformer architecture. If use string,
            choose from 'small', 'base', 'large', 'deit-tiny', 'deit-small'
            and 'deit-base'. If use dict, it should have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **feedforward_channels** (int): The hidden dimensions in
              feedforward modules.

            Defaults to 'base'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Defaults to True.
        avg_token (bool): Whether or not to use the mean patch token for
            classification. If True, the model will only take the average
            of all patch tokens. Defaults to False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        output_cls_token (bool): Whether output the cls_token. If set True,
            ``with_cls_token`` must be True. Defaults to True.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    # =========================================================================
    # arch_zoo: A "zoo" (collection) of pre-defined architecture configurations.
    # Each entry maps a name (like 'base') to a dict specifying:
    #   - embed_dims: dimension of each token vector
    #   - num_layers: how many Transformer encoder layers to stack
    #   - num_heads: how many parallel attention heads per layer
    #   - feedforward_channels: hidden dimension inside the FFN
    #
    # Example: ViT-Base has 768-dim tokens, 12 layers, 12 heads, 3072 FFN dim.
    # =========================================================================
    arch_zoo = {
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 768,
                'num_layers': 8,
                'num_heads': 8,
                'feedforward_channels': 768 * 3,
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 3072
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096
            }),
        **dict.fromkeys(
            ['h', 'huge'],
            {
                # The same as the implementation in MAE
                # <https://arxiv.org/abs/2111.06377>
                'embed_dims': 1280,
                'num_layers': 32,
                'num_heads': 16,
                'feedforward_channels': 5120
            }),
        **dict.fromkeys(
            ['eva-g', 'eva-giant'],
            {
                # The implementation in EVA
                # <https://arxiv.org/abs/2211.07636>
                'embed_dims': 1408,
                'num_layers': 40,
                'num_heads': 16,
                'feedforward_channels': 6144
            }),
        **dict.fromkeys(
            ['deit-t', 'deit-tiny'], {
                'embed_dims': 192,
                'num_layers': 12,
                'num_heads': 3,
                'feedforward_channels': 192 * 4
            }),
        **dict.fromkeys(
            ['deit-s', 'deit-small'], {
                'embed_dims': 384,
                'num_layers': 12,
                'num_heads': 6,
                'feedforward_channels': 384 * 4
            }),
        **dict.fromkeys(
            ['deit-b', 'deit-base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 768 * 4
            }),
    }

    # num_extra_tokens = 1 because ViT prepends 1 CLS token to the patch sequence.
    # Position embedding has (num_patches + 1) entries to include the CLS token position.
    num_extra_tokens = 1  # cls_token

    def __init__(self,
                 arch='base',             # Which architecture variant to use (see arch_zoo above)
                 img_size=224,            # Expected input image size (224x224 is standard for ImageNet)
                 patch_size=16,           # Each 16x16 pixel region becomes one token
                 in_channels=3,           # Number of image channels (3 = RGB)
                 out_indices=-1,          # Which layer(s) to output from (-1 = last layer)
                 drop_rate=0.,            # Dropout rate for regularization
                 drop_path_rate=0.,       # DropPath rate (stochastic depth — randomly skip layers)
                 qkv_bias=True,           # Whether Q/K/V projections have bias terms
                 norm_cfg=dict(type='LN', eps=1e-6),  # LayerNorm config
                 final_norm=True,         # Apply LayerNorm after the last encoder layer
                 with_cls_token=True,     # Whether to use the [CLS] token
                 avg_token=False,         # If True, average all patch tokens instead of using [CLS]
                 frozen_stages=-1,        # Freeze layers up to this index (-1 = don't freeze)
                 output_cls_token=True,   # Whether to include [CLS] token in the output
                 interpolate_mode='bicubic',  # How to resize position embeddings if image size changes
                 patch_cfg=dict(),        # Extra config for patch embedding (can override defaults)
                 layer_cfgs=dict(),       # Extra config for encoder layers (can override defaults)
                 pre_norm=False,          # Apply LayerNorm before the encoder (used by CLIP)
                 init_cfg=None):          # Weight initialization config (e.g., load pretrained)
        super(VisionTransformer, self).__init__(init_cfg)

        # --- Step 1: Resolve the architecture settings ---
        # If arch is a string like 'base', look it up in arch_zoo.
        # If arch is a custom dict, validate it has the required keys.
        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        # Store key dimensions used throughout the model
        self.embed_dims = self.arch_settings['embed_dims']    # e.g., 768 for ViT-Base
        self.num_layers = self.arch_settings['num_layers']    # e.g., 12 for ViT-Base
        self.img_size = to_2tuple(img_size)                   # Convert 224 -> (224, 224)

        # --- Step 2: Create the Patch Embedding layer ---
        # This layer uses a Conv2d to split the image into non-overlapping patches
        # and project each patch into an embed_dims-dimensional vector.
        # For a 224x224 image with patch_size=16: 224/16 = 14 patches per side -> 196 patches total
        # Output shape: (batch_size, 196, 768) for ViT-Base
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            bias=not pre_norm,  # disable bias if pre_norm is used(e.g., CLIP)
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size  # e.g., (14, 14)
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]  # e.g., 196

        # --- Step 3: Create the [CLS] token ---
        # The CLS token is a special learnable vector prepended to the patch sequence.
        # After passing through all layers, the CLS token's output is used as the
        # "image representation" for classification. Think of it as a "summary" token
        # that gathers information from all patches via attention.
        # Shape: (1, 1, embed_dims) — will be expanded to match batch size later
        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        # --- Step 4: Create Position Embeddings ---
        # Since Transformers have no built-in sense of order/position, we add
        # learnable position embeddings to each token so the model knows WHERE
        # each patch was in the original image.
        # Shape: (1, num_patches + 1, embed_dims) — the +1 is for the CLS token
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_extra_tokens,
                        self.embed_dims))
        # Register a hook to automatically resize pos_embed when loading weights
        # trained at a different image resolution
        self._register_load_state_dict_pre_hook(self._prepare_pos_embed)

        # Dropout applied after adding position embeddings
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # --- Step 5: Validate and convert out_indices ---
        # out_indices controls which layers' outputs we collect.
        # -1 means the last layer. Convert negative indices to positive.
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # --- Step 6: Build the stack of Transformer encoder layers ---
        # "Stochastic depth" linearly increases drop_path_rate from 0 to the max
        # across layers. Early layers have low drop rate, later layers have higher.
        # This helps regularize deeper layers more aggressively.
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers  # Same config for all layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],  # Linearly increasing drop rate per layer
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(TransformerEncoderLayer(**_layer_cfg))

        # --- Step 7: Optional pre-normalization (used by CLIP) ---
        self.frozen_stages = frozen_stages
        if pre_norm:
            _, norm_layer = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)
        else:
            norm_layer = nn.Identity()  # No-op if pre_norm is disabled
        self.add_module('pre_norm', norm_layer)

        # --- Step 8: Final normalization layer ---
        # Applied after the last encoder layer for stable features
        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        # --- Step 9: Optional average-token pooling ---
        # If avg_token=True, average all patch tokens instead of using [CLS]
        self.avg_token = avg_token
        if avg_token:
            self.norm2_name, norm2 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=2)
            self.add_module(self.norm2_name, norm2)

        # --- Step 10: Freeze early layers if requested ---
        # Freezing stops gradients from flowing into those layers, keeping
        # their weights fixed during training (useful for fine-tuning)
        if self.frozen_stages > 0:
            self._freeze_stages()

    @property
    def norm1(self):
        """Access the final LayerNorm by its stored name."""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """Access the average-token LayerNorm by its stored name."""
        return getattr(self, self.norm2_name)

    def init_weights(self):
        """Initialize model weights.
        If NOT loading pretrained weights, initialize position embeddings
        with a truncated normal distribution (values near 0, std=0.02).
        This gives a reasonable starting point for learning positions.
        """
        super(VisionTransformer, self).init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=0.02)

    def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
        """Hook that runs BEFORE loading a checkpoint's state_dict.

        Problem: A pretrained model might have been trained at a different image
        resolution (e.g., 224x224) than the current model (e.g., 384x384).
        Different resolutions produce different numbers of patches, so the
        position embedding sizes won't match.

        Solution: Interpolate (resize) the checkpoint's position embeddings
        to match the current model's expected number of patches.
        """
        name = prefix + 'pos_embed'
        if name not in state_dict.keys():
            return  # No pos_embed in checkpoint, nothing to do

        ckpt_pos_embed_shape = state_dict[name].shape
        if self.pos_embed.shape != ckpt_pos_embed_shape:
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            logger.info(
                f'Resize the pos_embed shape from {ckpt_pos_embed_shape} '
                f'to {self.pos_embed.shape}.')

            # Infer the spatial resolution of the checkpoint's patches
            # e.g., 197 tokens - 1 CLS token = 196 patches -> sqrt(196) = 14 -> (14, 14)
            ckpt_pos_embed_shape = to_2tuple(
                int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
            pos_embed_shape = self.patch_embed.init_out_size

            # Resize using bicubic interpolation
            state_dict[name] = resize_pos_embed(state_dict[name],
                                                ckpt_pos_embed_shape,
                                                pos_embed_shape,
                                                self.interpolate_mode,
                                                self.num_extra_tokens)

    @staticmethod
    def resize_pos_embed(*args, **kwargs):
        """Interface for backward-compatibility."""
        return resize_pos_embed(*args, **kwargs)

    def _freeze_stages(self):
        """Freeze (lock) certain parts of the model so they don't update during training.

        This is useful for transfer learning / fine-tuning: you keep the early
        layers fixed (they already learned good general features from pretraining)
        and only train the later layers on your specific task.
        """
        # Freeze position embedding — it won't be updated by the optimizer
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        # Set dropout to eval mode (disables dropout during frozen forward passes)
        self.drop_after_pos.eval()
        # Freeze the patch embedding layer
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # Freeze the CLS token
        self.cls_token.requires_grad = False
        # Freeze encoder layers up to frozen_stages
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        # If ALL layers are frozen, also freeze the final normalization
        if self.frozen_stages == len(self.layers) and self.final_norm:
            self.norm1.eval()
            for param in self.norm1.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward pass: image in -> features out.

        Args:
            x: Input image tensor of shape (batch_size, 3, H, W)
                e.g., (32, 3, 224, 224) for a batch of 32 RGB 224x224 images

        Returns:
            tuple: For Knowledge Distillation, each output element is:
                [[low_f, high_f, mid_token], cls_token]
                - low_f:     features from early layers (layers 0-1), shape (B, 2, num_patches, C)
                - high_f:    features from the last layer, shape (B, num_patches, C)
                - mid_token: CLS token from the middle layer, shape (B, C)
                - cls_token: CLS token from the final layer, shape (B, C)
        """
        B = x.shape[0]  # Batch size (number of images in this batch)

        # === STEP 1: Patch Embedding ===
        # Split the image into patches and project each to an embedding vector.
        # Input:  (B, 3, 224, 224)
        # Output: (B, 196, 768) for ViT-Base with 16x16 patches
        #         patch_resolution = (14, 14) — the grid of patches
        x, patch_resolution = self.patch_embed(x)

        # === STEP 2: Prepend [CLS] Token ===
        # Expand the single CLS token to match the batch size, then prepend it.
        # Before: x is (B, 196, 768)
        # After:  x is (B, 197, 768) — 1 CLS token + 196 patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # === STEP 3: Add Position Embeddings ===
        # Each of the 197 tokens gets a unique position vector added to it.
        # resize_pos_embed handles the case where the image resolution differs
        # from what the model was initialized with.
        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)

        # Apply dropout for regularization
        x = self.drop_after_pos(x)

        # Optional pre-normalization (used by CLIP models)
        x = self.pre_norm(x)

        # If not using CLS token, remove it before entering the encoder
        if not self.with_cls_token:
            x = x[:, 1:]  # Remove the first token (CLS)

        # === STEP 4: Pass Through All Transformer Encoder Layers ===
        outs = []
        for i, layer in enumerate(self.layers):
            # Run one encoder layer: Attention + FFN with residual connections
            x = layer(x)

            # Apply final LayerNorm after the very last layer
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            # ============================================================
            # KNOWLEDGE DISTILLATION (KD) FEATURE EXTRACTION
            # These blocks collect intermediate features for KD.
            # In KD, a large "teacher" model teaches a small "student" model
            # by sharing its internal features at different depths.
            # ============================================================

            # --- Collect LOW-LEVEL features (from layers 0 and 1) ---
            # Low-level features capture basic patterns like edges and textures.
            # x[:, 1:] means "all patch tokens, excluding the CLS token at index 0"
            if i in [1]:
                # Layer 1: grab patch tokens and stack with layer 0's features
                low_f_s = x[:, 1:]  # Shape: (B, num_patches, C)
                # Concatenate along a new dimension to build: (B, 2, num_patches, C)
                low_f = torch.cat((low_f, low_f_s.unsqueeze(1)), dim=1)
            elif i == 0:
                # Layer 0: first low-level feature, add a dimension for stacking
                low_f = x[:, 1:]          # Shape: (B, num_patches, C)
                low_f = low_f.unsqueeze(1) # Shape: (B, 1, num_patches, C)

            # --- Collect HIGH-LEVEL features (from the last layer) ---
            # High-level features capture abstract/semantic information.
            if i == len(self.layers) - 1:
                high_f = x[:, 1:]  # Shape: (B, num_patches, C)

            # --- Collect MID-LEVEL CLS token (from the middle layer) ---
            # The CLS token at the halfway point gives a "mid-depth" summary.
            # For a 12-layer model: int(0.5 * 12) - 1 = 5, so layer index 5.
            if i == int(0.5 * len(self.layers)) - 1:
                mid_token = x[:, 0]  # Shape: (B, C) — just the CLS token

            # === STEP 5: Collect Output at Specified Layer Indices ===
            if i in self.out_indices:
                B, _, C = x.shape  # (batch_size, num_tokens, channels)

                if self.with_cls_token:
                    # Separate patch tokens from CLS token
                    # Reshape patch tokens back to spatial grid: (B, C, H_patches, W_patches)
                    patch_token = x[:, 1:].reshape(B, *patch_resolution, C)
                    patch_token = patch_token.permute(0, 3, 1, 2)  # (B, C, H, W)
                    cls_token = x[:, 0]  # (B, C)
                else:
                    patch_token = x.reshape(B, *patch_resolution, C)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = None

                # Optional: average all patch tokens for classification
                if self.avg_token:
                    patch_token = patch_token.permute(0, 2, 3, 1)
                    patch_token = patch_token.reshape(
                        B, patch_resolution[0] * patch_resolution[1],
                        C).mean(dim=1)  # Global average pooling over patches
                    patch_token = self.norm2(patch_token)

                if self.output_cls_token:
                    # *** KD OUTPUT FORMAT ***
                    # Instead of the standard [patch_token, cls_token] output,
                    # we return intermediate features for knowledge distillation:
                    #   low_f:     early-layer patch features (layers 0-1)
                    #   high_f:    last-layer patch features
                    #   mid_token: CLS token from the middle layer
                    #   cls_token: CLS token from the final layer (used for classification)
                    out = [[low_f, high_f, mid_token], cls_token]
                else:
                    out = patch_token
                outs.append(out)

        # Return as a tuple (standard format for multi-output backbones)
        return tuple(outs)
