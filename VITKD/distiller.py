from abc import ABCMeta, abstractmethod
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcls.models import build_classifier
from mmcls.registry import MODELS
from mmcls.structures import ClsDataSample

from mmengine.config import Config
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement
from mmengine.runner.checkpoint import load_checkpoint, _load_checkpoint, load_state_dict

@MODELS.register_module()
class ClassificationDistiller(BaseModel, metaclass=ABCMeta):
    """
    Knowledge Distillation Orchestrator for Vision Transformers
    
    This class manages the entire knowledge distillation training process:
    1. Maintains both teacher and student models
    2. Extracts features from both models
    3. Computes multiple distillation losses (feature-based and logit-based)
    4. Combines all losses for backpropagation
    
    The teacher model is frozen (no gradient updates), while the student
    and distillation modules are trainable.
    """
    
    def __init__(self,
                 teacher_cfg,           # Path to teacher config file
                 student_cfg,           # Path to student config file
                 is_vit=False,          # Whether models are Vision Transformers
                 use_logit=False,       # Whether to use logit-based distillation
                 sd=False,              # Whether to use self-distillation
                 distill_cfg=None,      # Configuration for distillation losses
                 teacher_pretrained=None,  # Path to pretrained teacher weights
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        
        # ═══════════════════════════════════════════════════════════════════════
        # SETUP DATA PREPROCESSOR
        # ═══════════════════════════════════════════════════════════════════════
        # Handles image normalization, augmentation, etc.
        if data_preprocessor is None:
            data_preprocessor = {}
        data_preprocessor.setdefault('type', 'mmcls.ClsDataPreprocessor')

        # Add batch augmentations (Mixup, CutMix) if specified in train_cfg
        if train_cfg is not None and 'augments' in train_cfg:
            data_preprocessor['batch_augments'] = train_cfg

        # Initialize base model with preprocessor
        super(ClassificationDistiller, self).__init__(
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        # ═══════════════════════════════════════════════════════════════════════
        # BUILD AND FREEZE TEACHER MODEL
        # ═══════════════════════════════════════════════════════════════════════
        # The teacher is a larger, pretrained model that guides the student
        
        # Load teacher configuration and build model
        # Example: DeiT-Small (80.69% accuracy, 22M parameters)
        self.teacher = build_classifier((Config.fromfile(teacher_cfg)).model)
        self.teacher_pretrained = teacher_pretrained
        
        # Set teacher to evaluation mode (disable dropout, batch norm updates, etc.)
        self.teacher.eval()
        
        # Freeze all teacher parameters - no gradient computation or updates
        # This saves memory and computation since teacher doesn't learn
        for param in self.teacher.parameters():
            param.requires_grad = False

        # ═══════════════════════════════════════════════════════════════════════
        # BUILD STUDENT MODEL (Trainable)
        # ═══════════════════════════════════════════════════════════════════════
        # The student is a smaller model that learns from the teacher
        # Example: DeiT-Tiny (baseline 74.42% accuracy, 5M parameters)
        self.student = build_classifier((Config.fromfile(student_cfg)).model)

        # ═══════════════════════════════════════════════════════════════════════
        # BUILD DISTILLATION LOSS MODULES
        # ═══════════════════════════════════════════════════════════════════════
        # Create a dictionary to hold all enabled distillation losses
        self.distill_cfg = distill_cfg
        self.distill_losses = nn.ModuleDict()
        
        if self.distill_cfg is not None:
            # Iterate through distillation configuration
            for item_loc in distill_cfg:
                for item_loss in item_loc.methods:
                    loss_name = item_loss.name  # e.g., 'loss_vitkd', 'loss_nkd'
                    use_this = item_loss.use_this  # Boolean flag to enable/disable
                    
                    if use_this:
                        # Build and register the loss module
                        # Example: ViTKDLoss(D_S=192, D_T=384, α=3e-5, β=3e-6, λ=0.5)
                        self.distill_losses[loss_name] = MODELS.build(item_loss)

        # ═══════════════════════════════════════════════════════════════════════
        # STORE CONFIGURATION FLAGS
        # ═══════════════════════════════════════════════════════════════════════
        self.is_vit = is_vit          # True for ViT-based models (DeiT, Swin, etc.)
        self.sd = sd                  # True for self-distillation (no teacher needed)
        self.use_logit = use_logit    # True to enable logit-based distillation

    def init_weights(self):
        """
        Initialize model weights.
        - Load pretrained weights for teacher
        - Initialize student weights (random or pretrained)
        """
        if self.teacher_pretrained is not None:
            # Load pretrained teacher weights from checkpoint
            load_checkpoint(self.teacher, self.teacher_pretrained, map_location='cpu')
        
        # Initialize student weights (calls student's init_weights method)
        self.student.init_weights()

    def forward(self,
                inputs: torch.Tensor,              # Input images [B, 3, 224, 224]
                data_samples: Optional[List[BaseDataElement]] = None,  # Labels and metadata
                mode: str = 'tensor'):             # Mode: 'loss', 'predict', or 'tensor'
        """
        Forward pass - routes to appropriate method based on mode.
        
        Args:
            inputs: Batch of images, shape [B, 3, H, W]
            data_samples: List of data samples containing labels and metadata
            mode: Operating mode
                - 'tensor': Feature extraction only
                - 'loss': Training mode (compute losses)
                - 'predict': Inference mode (generate predictions)
        
        Returns:
            Depends on mode:
                - 'tensor': Feature tensor
                - 'loss': Dictionary of losses
                - 'predict': Predictions
        """
        if mode == 'tensor':
            # Feature extraction mode - just extract features without head
            feats = self.student.extract_feat(inputs)
            return self.student.head(feats) if self.student.with_head else feats
        
        elif mode == 'loss':
            # Training mode - compute all losses
            return self.loss(inputs, data_samples)
        
        elif mode == 'predict':
            # Inference mode - generate predictions using student only
            return self.student.predict(inputs, data_samples)
        
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def loss(self, 
             inputs: torch.Tensor,              # Input images [B, 3, 224, 224]
             data_samples: List[ClsDataSample]  # Ground truth labels and metadata
             ) -> dict:
        """
        Compute all losses for knowledge distillation training.
        
        This is the main training logic that:
        1. Runs student forward pass and computes classification loss
        2. Runs teacher forward pass (no gradients)
        3. Computes distillation losses (ViTKD, NKD, etc.)
        4. Returns dictionary of all losses
        
        Args:
            inputs: Batch of images, shape [B, 3, 224, 224]
            data_samples: List of ClsDataSample objects containing:
                - gt_label: Ground truth labels (one-hot or class indices)
                - Other metadata (image paths, etc.)
        
        Returns:
            Dictionary of losses, e.g.:
                {
                    'ori_loss': 2.34,       # Classification loss
                    'loss_vitkd': 0.15,     # Feature distillation
                    'loss_nkd': 0.08        # Logit distillation
                }
            MMEngine will automatically sum these and call .backward()
        """
        
        # ═══════════════════════════════════════════════════════════════════════
        # EXTRACT GROUND TRUTH LABELS
        # ═══════════════════════════════════════════════════════════════════════
        # Labels can be in two formats:
        # 1. One-hot scores (after Mixup/CutMix): [B, num_classes]
        # 2. Class indices: [B]
        
        if 'score' in data_samples[0].gt_label:
            # Batch augmentation converted labels to one-hot format
            # Stack all samples: List of [num_classes] → [B, num_classes]
            gt_label = torch.stack([i.gt_label.score for i in data_samples])
        else:
            # Standard class indices
            # Concatenate all samples: List of [1] → [B]
            gt_label = torch.cat([i.gt_label.label for i in data_samples])

        # ═══════════════════════════════════════════════════════════════════════
        # STUDENT FORWARD PASS (With Gradients)
        # ═══════════════════════════════════════════════════════════════════════
        # Extract features from student's backbone (all transformer layers)
        # For ViT, returns: [[low_f, high_f, mid_token], cls_token]
        # Where:
        #   low_f: Features from layers 0-1, shape [B, 2, 196, 192]
        #   high_f: Features from last layer, shape [B, 196, 192]
        #   mid_token: CLS token from middle layer (for self-distillation)
        #   cls_token: Final CLS token, shape [B, 192]
        fea_s = self.student.extract_feat(inputs, stage='backbone')

        # Prepare features for classification head
        x = fea_s
        
        # Apply neck if present (usually not used in ViT)
        if self.student.with_neck:
            x = self.student.neck(x)
        
        # Apply pre-logits processing (e.g., extract CLS token, apply LayerNorm)
        if self.student.with_head and hasattr(self.student.head, 'pre_logits'):
            x = self.student.head.pre_logits(x)

        # Generate classification logits
        if self.is_vit:
            # For ViT models, classification head is in head.layers.head
            # Shape: [B, 192] → [B, 1000] for ImageNet
            logit_s = self.student.head.layers.head(x)
        else:
            # For CNN models, classification head is in head.fc
            logit_s = self.student.head.fc(x)
        
        # Compute standard classification loss (cross-entropy with label smoothing)
        # Also accounts for Mixup/CutMix if used
        loss = self.student.head._get_loss(logit_s, data_samples)

        # Initialize loss dictionary with classification loss
        s_loss = dict()
        for key in loss.keys():
            # Prefix with 'ori_' to distinguish from distillation losses
            s_loss['ori_'+key] = loss[key]

        # ═══════════════════════════════════════════════════════════════════════
        # TEACHER-STUDENT KNOWLEDGE DISTILLATION
        # ═══════════════════════════════════════════════════════════════════════
        # Only run if not using self-distillation
        if not self.sd:
            
            # ═══════════════════════════════════════════════════════════════════
            # TEACHER FORWARD PASS (No Gradients)
            # ═══════════════════════════════════════════════════════════════════
            # Wrap in no_grad() to disable gradient computation for teacher
            # This saves memory and speeds up training
            with torch.no_grad():
                # Extract features from teacher's backbone
                # Returns: [[low_f, high_f], cls_token]
                # Where:
                #   low_f: Features from layers 0-1, shape [B, 2, 196, 384]
                #   high_f: Features from last layer, shape [B, 196, 384]
                #   cls_token: Final CLS token, shape [B, 384]
                fea_t = self.teacher.extract_feat(inputs, stage='backbone')
                
                # If using logit-based distillation, also get teacher's predictions
                if self.use_logit:
                    if self.is_vit:
                        # ViT teacher
                        logit_t = self.teacher.head.layers.head(
                            self.teacher.head.pre_logits(fea_t))
                    else:
                        # CNN teacher
                        logit_t = self.teacher.head.fc(
                            self.teacher.head.pre_logits(
                                self.teacher.neck(fea_t)))

            # ═══════════════════════════════════════════════════════════════════
            # COMPUTE DISTILLATION LOSSES
            # ═══════════════════════════════════════════════════════════════════
            # Get list of all enabled distillation losses
            all_keys = self.distill_losses.keys()
            
            # ───────────────────────────────────────────────────────────────────
            # ViTKD Loss (Feature-based distillation)
            # ───────────────────────────────────────────────────────────────────
            if 'loss_vitkd' in all_keys:
                loss_name = 'loss_vitkd'
                # Pass shallow and deep features to ViTKD loss
                # fea_s[-1][0] = [low_f_s, high_f_s, mid_token_s]
                # fea_t[-1][0] = [low_f_t, high_f_t]
                # Returns: α*L_lr + β*L_gen
                s_loss[loss_name] = self.distill_losses[loss_name](
                    fea_s[-1][0],  # Student features
                    fea_t[-1][0]   # Teacher features
                )

            # ───────────────────────────────────────────────────────────────────
            # SRRL Loss (Spatial-wise feature distillation)
            # ───────────────────────────────────────────────────────────────────
            if ('loss_srrl' in all_keys) and self.use_logit:
                loss_name = 'loss_srrl'
                # Align student features using SRRL's connector
                fea_s_align = self.distill_losses[loss_name].Connectors(fea_s[-1])
                # Get logits using teacher's head on aligned student features
                logit_st = self.teacher.head.fc(
                    self.teacher.head.pre_logits(
                        self.teacher.neck(fea_s_align)))
                # Compute SRRL loss
                s_loss[loss_name] = self.distill_losses[loss_name](
                    fea_s_align, fea_t[-1], logit_st, logit_t)

            # ───────────────────────────────────────────────────────────────────
            # MGD Loss (Masked Generative Distillation)
            # ───────────────────────────────────────────────────────────────────
            if 'loss_mgd' in all_keys:
                loss_name = 'loss_mgd'
                s_loss[loss_name] = self.distill_losses[loss_name](
                    fea_s[-1], fea_t[-1])

            # ───────────────────────────────────────────────────────────────────
            # WSLD Loss (Weighted Soft Label Distillation)
            # ───────────────────────────────────────────────────────────────────
            if ('loss_wsld' in all_keys) and self.use_logit:
                loss_name = 'loss_wsld'
                s_loss[loss_name] = self.distill_losses[loss_name](
                    logit_s, logit_t, gt_label)

            # ───────────────────────────────────────────────────────────────────
            # DKD Loss (Decoupled Knowledge Distillation)
            # ───────────────────────────────────────────────────────────────────
            if ('loss_dkd' in all_keys) and self.use_logit:
                loss_name = 'loss_dkd'
                s_loss[loss_name] = self.distill_losses[loss_name](
                    logit_s, logit_t, gt_label)

            # ───────────────────────────────────────────────────────────────────
            # KD Loss (Classic Knowledge Distillation)
            # ───────────────────────────────────────────────────────────────────
            if ('loss_kd' in all_keys) and self.use_logit:
                loss_name = 'loss_kd'
                # KD also returns alpha to reweight original loss
                ori_alpha, s_loss[loss_name] = self.distill_losses[loss_name](
                    logit_s, logit_t)
                # Adjust original loss weight
                s_loss['ori_loss'] = ori_alpha * s_loss['ori_loss']

            # ───────────────────────────────────────────────────────────────────
            # NKD Loss (Normalized Knowledge Distillation)
            # ───────────────────────────────────────────────────────────────────
            if ('loss_nkd' in all_keys) and self.use_logit:
                loss_name = 'loss_nkd'
                # NKD is current state-of-the-art for logit distillation
                s_loss[loss_name] = self.distill_losses[loss_name](
                    logit_s, logit_t, gt_label)

        # ═══════════════════════════════════════════════════════════════════════
        # SELF-KNOWLEDGE DISTILLATION
        # ═══════════════════════════════════════════════════════════════════════
        # Student learns from its own intermediate layers (no teacher needed)
        if self.sd:
            all_keys = self.distill_losses.keys()
            
            # ───────────────────────────────────────────────────────────────────
            # USKD Loss (Unified Self-Knowledge Distillation)
            # ───────────────────────────────────────────────────────────────────
            if 'loss_uskd' in all_keys:
                loss_name = 'loss_uskd'
                
                if self.is_vit:
                    # For ViT: Use middle layer CLS token
                    # fea_s[-1][0][2] = mid_token from layer 5
                    fea_mid = fea_s[-1][0][2]
                else:
                    # For CNN: Use features from second-to-last layer
                    fea_mid = self.student.neck(fea_s[-2])
                
                # Compute USKD loss using middle features, final logits, and labels
                s_loss[loss_name] = self.distill_losses[loss_name](
                    fea_mid, logit_s, gt_label)

        # ═══════════════════════════════════════════════════════════════════════
        # RETURN ALL LOSSES
        # ═══════════════════════════════════════════════════════════════════════
        # MMEngine will automatically:
        # 1. Sum all values in the dictionary
        # 2. Call total_loss.backward()
        # 3. Update student parameters (and distillation module parameters)
        # 
        # Example return value:
        # {
        #     'ori_loss': 2.34,       # Cross-entropy with label smoothing
        #     'loss_vitkd': 0.15,     # ViTKD feature distillation
        #     'loss_nkd': 0.08        # NKD logit distillation
        # }
        # Total loss = 2.34 + 0.15 + 0.08 = 2.57
        return s_loss