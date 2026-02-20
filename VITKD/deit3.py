# Copyright (c) OpenMMLab. All rights reserved.
# =============================================================================
# This file implements DeiT III (Data-efficient Image Transformers III),
# which is an improved version of Vision Transformer (ViT).
#
# Paper: "DeiT III: Revenge of the ViT" (https://arxiv.org/pdf/2204.07118.pdf)
#
# KEY DIFFERENCES from standard ViT (vision_transformer.py):
#   1. Uses "LayerScale" — learnable per-channel scaling factors that multiply
#      the output of attention and FFN blocks. This stabilizes training,
#      especially for deeper/larger models.
#   2. The CLS token is added AFTER position embeddings (not before).
#      In standard ViT, CLS token is prepended first, then pos_embed is added.
#      In DeiT3, pos_embed is added to patches first, then CLS token is prepended.
#      This means pos_embed does NOT include a slot for the CLS token.
#
# This version is also MODIFIED for Knowledge Distillation (KD), extracting
# features from early and late layers for the student to learn from.
# =============================================================================

from typing import Sequence  # For type-hinting lists/tuples

import numpy as np            # Numerical ops (linspace for drop rates)
import torch                  # Core PyTorch
from mmcv.cnn import Linear, build_activation_layer, build_norm_layer  # NN building helpers
from mmcv.cnn.bricks.drop import build_dropout          # Dropout builder
from mmcv.cnn.bricks.transformer import PatchEmbed       # Splits image into patch embeddings
from mmengine.model import BaseModule, ModuleList, Sequential  # Model base classes
from mmengine.utils import deprecated_api_warning        # Warns about old parameter names
from torch import nn                                     # Neural network modules

from mmcls.registry import MODELS                                        # Model registry
from ..utils import LayerScale, MultiheadAttention, resize_pos_embed, to_2tuple  # Utilities
from .vision_transformer import VisionTransformer        # Parent class that DeiT3 inherits from


# =============================================================================
# DeiT3FFN: Feed-Forward Network with LayerScale
#
# This is almost the same as a standard FFN (two linear layers with activation),
# but adds "LayerScale": a learnable per-channel scaling vector (gamma2) that
# multiplies the FFN output before the residual connection.
#
# Why LayerScale? In deep transformers, the outputs of attention and FFN can be
# too large and destabilize training. LayerScale initializes these multipliers
# to very small values (e.g., 1e-4), so initially the residual connections
# dominate, and the model gradually learns to trust the layer outputs more.
#
# Standard FFN:   output = x + FFN(LayerNorm(x))
# DeiT3 FFN:      output = x + LayerScale(FFN(LayerNorm(x)))
# =============================================================================


class DeiT3FFN(BaseModule):
    """FFN for DeiT3.

    The differences between DeiT3FFN & FFN:
        1. Use LayerScale.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        use_layer_scale (bool): Whether to use layer_scale in
            DeiT3FFN. Defaults to True.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    @deprecated_api_warning(
        {
            'dropout': 'ffn_drop',       # Old param name 'dropout' -> new name 'ffn_drop'
            'add_residual': 'add_identity'  # Old 'add_residual' -> new 'add_identity'
        },
        cls_name='FFN')
    def __init__(self,
                 embed_dims=256,           # Input/output feature dimension
                 feedforward_channels=1024,# Hidden layer size (expansion)
                 num_fcs=2,                # Number of fully-connected layers (must be >= 2)
                 act_cfg=dict(type='ReLU', inplace=True),  # Activation function
                 ffn_drop=0.,              # Dropout probability inside the FFN
                 dropout_layer=None,       # Extra dropout for the residual shortcut
                 add_identity=True,        # Whether to add the residual (skip) connection
                 use_layer_scale=True,     # Whether to apply LayerScale (the key DeiT3 innovation)
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)  # Create the activation function (e.g., ReLU)

        # --- Build the FFN layers ---
        # Typical structure for num_fcs=2:
        #   Linear(embed_dims -> feedforward_channels) -> Activation -> Dropout
        #   Linear(feedforward_channels -> embed_dims) -> Dropout
        # This is the "expand then compress" pattern: features are projected to
        # a wider space for richer transformations, then projected back.
        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels), self.activate,
                    nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))  # Final projection back
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)

        # Dropout for the residual/shortcut path
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

        # --- LayerScale (DeiT3's key addition) ---
        # gamma2 is a learnable vector of shape (embed_dims,), initialized to
        # small values. It scales the FFN output channel-wise before adding
        # the residual connection. This stabilizes training of deep models.
        if use_layer_scale:
            self.gamma2 = LayerScale(embed_dims)
        else:
            self.gamma2 = nn.Identity()  # No scaling if disabled

    @deprecated_api_warning({'residual': 'identity'}, cls_name='FFN')
    def forward(self, x, identity=None):
        """Forward function for DeiT3 FFN.

        Args:
            x: Input tensor of shape (B, num_tokens, embed_dims)
            identity: Optional residual tensor. If None, uses x itself.

        Returns:
            Tensor: identity + LayerScale(FFN(x))
            The LayerScale (gamma2) is what makes this different from standard FFN.
        """
        out = self.layers(x)       # Pass through the MLP layers
        out = self.gamma2(out)      # Apply LayerScale: multiply by learned per-channel weights
        if not self.add_identity:
            return self.dropout_layer(out)  # No residual connection
        if identity is None:
            identity = x            # Use input as the residual
        return identity + self.dropout_layer(out)  # Residual connection: identity + scaled FFN output


# =============================================================================
# DeiT3TransformerEncoderLayer: One encoder layer with LayerScale.
#
# This is identical to the standard TransformerEncoderLayer (Attention + FFN),
# except that BOTH the attention and FFN outputs are scaled by LayerScale
# before being added to the residual.
#
# Standard ViT layer:  x = x + Attention(Norm(x))
#                       x = x + FFN(Norm(x))
#
# DeiT3 layer:         x = x + LayerScale(Attention(Norm(x)))
#                       x = x + LayerScale(FFN(Norm(x)))
# =============================================================================
class DeiT3TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in DeiT3.

    The differences between DeiT3TransformerEncoderLayer &
    TransformerEncoderLayer:
        1. Use LayerScale.

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
        use_layer_scale (bool): Whether to use layer_scale in
            DeiT3TransformerEncoderLayer. Defaults to True.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,          # Dimension of each token's feature vector
                 num_heads,           # Number of attention heads
                 feedforward_channels,# Hidden size in FFN
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 use_layer_scale=True,  # <<< DeiT3-specific: enable LayerScale
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(DeiT3TransformerEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims

        # --- LayerNorm before attention ---
        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        # --- Multi-Head Self-Attention (with LayerScale inside) ---
        # The `use_layer_scale=True` parameter tells the attention module
        # to apply LayerScale to its output before the residual addition.
        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias,
            use_layer_scale=use_layer_scale)  # <<< LayerScale applied to attention

        # --- LayerNorm before FFN ---
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        # --- DeiT3 FFN (with LayerScale inside) ---
        # Uses DeiT3FFN instead of the standard FFN, which also has LayerScale.
        self.ffn = DeiT3FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            use_layer_scale=use_layer_scale)  # <<< LayerScale applied to FFN

    @property
    def norm1(self):
        """Access LayerNorm #1 by its stored name."""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """Access LayerNorm #2 by its stored name."""
        return getattr(self, self.norm2_name)

    def init_weights(self):
        """Initialize FFN linear layers with Xavier uniform weights
        and small-valued biases for stable training."""
        super(DeiT3TransformerEncoderLayer, self).init_weights()
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        """Forward pass: same structure as standard ViT encoder layer.

        x = x + LayerScale(Attention(LayerNorm(x)))
        x = x + LayerScale(FFN(LayerNorm(x)))

        The LayerScale is applied INSIDE self.attn and self.ffn.
        """
        # Self-attention with residual connection (LayerScale is inside self.attn)
        x = x + self.attn(self.norm1(x))
        # FFN with residual connection (LayerScale is inside self.ffn via gamma2)
        x = self.ffn(self.norm2(x), identity=x)
        return x


# =============================================================================
# DeiT3: The full DeiT III backbone, inheriting from VisionTransformer.
#
# KEY DIFFERENCES from the parent VisionTransformer:
#   1. Uses DeiT3TransformerEncoderLayer (with LayerScale) instead of
#      TransformerEncoderLayer.
#   2. CLS token is added AFTER position embeddings, not before.
#      - ViT:   [CLS + patches] + pos_embed -> encoder
#      - DeiT3: patches + pos_embed -> [CLS + result] -> encoder
#      Because of this, num_extra_tokens = 0 (pos_embed only covers patches).
#   3. Has its own arch_zoo with slightly different size configurations
#      (includes 'medium' variant, no 'deit-*' prefixed variants).
#   4. No pre_norm, no avg_token, no frozen_stages (simpler interface).
# =============================================================================
@MODELS.register_module()  # Register so configs can reference 'DeiT3'
class DeiT3(VisionTransformer):
    """DeiT3 backbone.

    A PyTorch implement of : `DeiT III: Revenge of the ViT
    <https://arxiv.org/pdf/2204.07118.pdf>`_

    The differences between DeiT3 & VisionTransformer:

    1. Use LayerScale.
    2. Concat cls token after adding pos_embed.

    Args:
        arch (str | dict): DeiT3 architecture. If use string,
            choose from 'small', 'base', 'medium', 'large' and 'huge'.
            If use dict, it should have below keys:

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
        output_cls_token (bool): Whether output the cls_token. If set True,
            ``with_cls_token`` must be True. Defaults to True.
        use_layer_scale (bool): Whether to use layer_scale in  DeiT3.
            Defaults to True.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    # --- DeiT3's own architecture configurations ---
    # Note: includes 'medium' (m) which standard ViT doesn't have
    arch_zoo = {
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 384,
                'num_layers': 12,
                'num_heads': 6,
                'feedforward_channels': 1536,
            }),
        **dict.fromkeys(
            ['m', 'medium'], {              # <<< DeiT3-specific size variant
                'embed_dims': 512,
                'num_layers': 12,
                'num_heads': 8,
                'feedforward_channels': 2048,
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
            ['h', 'huge'], {
                'embed_dims': 1280,
                'num_layers': 32,
                'num_heads': 16,
                'feedforward_channels': 5120
            }),
    }

    # IMPORTANT DIFFERENCE from ViT:
    # num_extra_tokens = 0 because DeiT3 adds CLS token AFTER position embedding.
    # This means the pos_embed tensor only has slots for patch tokens (not CLS).
    # In ViT, num_extra_tokens = 1 because CLS is added before pos_embed.
    num_extra_tokens = 0

    def __init__(self,
                 arch='base',
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 with_cls_token=True,
                 output_cls_token=True,
                 use_layer_scale=True,     # <<< DeiT3-specific: enable LayerScale
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 init_cfg=None):
        # NOTE: Calls BaseBackbone.__init__ (grandparent), NOT VisionTransformer.__init__.
        # This is because DeiT3 overrides the entire setup logic to make
        # key changes (different layer type, different CLS token placement).
        super(VisionTransformer, self).__init__(init_cfg)

        # --- Resolve architecture settings (same pattern as ViT) ---
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

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.img_size = to_2tuple(img_size)

        # --- Patch Embedding (same as ViT) ---
        # Splits the image into patches and projects each to embed_dims
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # --- CLS Token ---
        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        # --- Position Embedding ---
        # IMPORTANT DIFFERENCE FROM ViT:
        # pos_embed shape is (1, num_patches, embed_dims) -- NO extra token slot!
        # In ViT it's (1, num_patches + 1, embed_dims) with +1 for CLS token.
        # DeiT3 adds CLS token AFTER position embedding, so it doesn't need
        # a position for CLS.
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.embed_dims))
        self._register_load_state_dict_pre_hook(self._prepare_pos_embed)

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # --- Validate output indices ---
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

        # --- Build encoder layers (using DeiT3TransformerEncoderLayer, NOT standard) ---
        dpr = np.linspace(0, drop_path_rate, self.num_layers)  # Stochastic depth schedule

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg,
                use_layer_scale=use_layer_scale)  # <<< Pass LayerScale flag
            _layer_cfg.update(layer_cfgs[i])
            # Uses DeiT3TransformerEncoderLayer instead of TransformerEncoderLayer
            self.layers.append(DeiT3TransformerEncoderLayer(**_layer_cfg))

        # --- Final normalization ---
        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

    def forward(self, x):
        """Forward pass for DeiT3.

        The key difference from ViT's forward:
          - ViT:   CLS token prepended FIRST, then pos_embed added
          - DeiT3: pos_embed added FIRST (to patches only), then CLS prepended

        Args:
            x: Input image tensor, shape (B, 3, H, W)

        Returns:
            tuple: For KD, each output element is:
                [[low_f, high_f], cls_token]
                - low_f:     features from early layers (layers 0-1), shape (B, 2, num_patches, C)
                - high_f:    features from the last layer, shape (B, num_patches, C)
                - cls_token: CLS token from the final layer, shape (B, C)
                (Note: DeiT3 does NOT extract mid_token, unlike ViT)
        """
        B = x.shape[0]  # Batch size

        # === STEP 1: Patch Embedding ===
        # Split image into patches -> (B, num_patches, embed_dims)
        x, patch_resolution = self.patch_embed(x)

        # === STEP 2: Add Position Embeddings to patches FIRST ===
        # DeiT3 difference: pos_embed is added BEFORE CLS token is prepended.
        # pos_embed shape is (1, num_patches, embed_dims) — no CLS slot.
        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)  # num_extra_tokens = 0
        x = self.drop_after_pos(x)

        # === STEP 3: THEN prepend the CLS token ===
        # Now x goes from (B, 196, 768) to (B, 197, 768)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Optionally remove CLS token if not used
        if not self.with_cls_token:
            x = x[:, 1:]

        # === STEP 4: Pass through all DeiT3 encoder layers ===
        outs = []
        for i, layer in enumerate(self.layers):
            # Run one DeiT3 encoder layer (with LayerScale inside)
            x = layer(x)

            # Apply final LayerNorm after the very last layer
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            # ============================================================
            # KNOWLEDGE DISTILLATION FEATURE EXTRACTION
            # Same concept as ViT, but DeiT3 does NOT extract mid_token.
            # ============================================================

            # --- Collect LOW-LEVEL features (layers 0 and 1) ---
            # x[:, 1:] = all patch tokens (excluding CLS at index 0)
            if i in [1]:
                # Layer 1: stack with layer 0's features
                low_f_s = x[:, 1:]  # (B, num_patches, C)
                low_f = torch.cat((low_f, low_f_s.unsqueeze(1)), dim=1)  # (B, 2, num_patches, C)
            elif i == 0:
                # Layer 0: initialize low-level features
                low_f = x[:, 1:]           # (B, num_patches, C)
                low_f = low_f.unsqueeze(1)  # (B, 1, num_patches, C)

            # --- Collect HIGH-LEVEL features (last layer) ---
            if i == len(self.layers) - 1:
                high_f = x[:, 1:]  # (B, num_patches, C)

            # === STEP 5: Collect output at specified layer indices ===
            if i in self.out_indices:
                B, _, C = x.shape
                if self.with_cls_token:
                    # Reshape patch tokens to spatial grid
                    patch_token = x[:, 1:].reshape(B, *patch_resolution, C)
                    patch_token = patch_token.permute(0, 3, 1, 2)  # (B, C, H, W)
                    cls_token = x[:, 0]  # (B, C)
                else:
                    patch_token = x.reshape(B, *patch_resolution, C)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = None
                if self.output_cls_token:
                    # *** KD OUTPUT FORMAT ***
                    # Returns [low_f, high_f] (no mid_token) + cls_token
                    out = [[low_f, high_f], cls_token]
                else:
                    out = patch_token
                outs.append(out)

        return tuple(outs)
