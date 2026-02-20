import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.registry import MODELS

@MODELS.register_module()
class ViTKDLoss(nn.Module):
    """
    ViTKD (Vision Transformer Knowledge Distillation) Loss Module
    
    This loss function implements feature-based knowledge distillation for Vision Transformers.
    It treats shallow and deep layers differently:
    - Shallow layers (0-1): Direct mimicking via MSE loss
    - Deep layer (last): Masked generation via convolutional reconstruction
    
    Paper: "ViTKD: Practical Guidelines for ViT feature knowledge distillation"
    """

    def __init__(self,
                 name,
                 use_this,
                 student_dims,      # Student's embedding dimension (e.g., 192 for DeiT-Tiny)
                 teacher_dims,      # Teacher's embedding dimension (e.g., 384 for DeiT-Small)
                 alpha_vitkd=0.00003,   # Weight for shallow layer mimicking loss
                 beta_vitkd=0.000003,   # Weight for deep layer generation loss
                 lambda_vitkd=0.5,      # Masking ratio (0.5 = mask 50% of tokens)
                 ):
        super(ViTKDLoss, self).__init__()
        
        # Store hyperparameters
        self.alpha_vitkd = alpha_vitkd      # Scales the mimicking loss (default: 3e-5)
        self.beta_vitkd = beta_vitkd        # Scales the generation loss (default: 3e-6)
        self.lambda_vitkd = lambda_vitkd    # Controls how many tokens to mask (default: 0.5)
    
        # ═══════════════════════════════════════════════════════════════════════
        # ALIGNMENT LAYERS: Match student's dimension to teacher's dimension
        # ═══════════════════════════════════════════════════════════════════════
        
        if student_dims != teacher_dims:
            # For shallow layers: Need separate linear layers for layer 0 and layer 1
            # Each transforms student features from D_student → D_teacher
            # Example: [B, N, 192] → [B, N, 384]
            self.align2 = nn.ModuleList([
                nn.Linear(student_dims, teacher_dims, bias=True)  # For layer 0
                for i in range(2)])                                # For layer 1
            
            # For deep layer: Single linear layer for the last layer
            # Transforms [B, N, 192] → [B, N, 384]
            self.align = nn.Linear(student_dims, teacher_dims, bias=True)
        else:
            # If dimensions already match, no alignment needed
            self.align2 = None
            self.align = None

        # ═══════════════════════════════════════════════════════════════════════
        # LEARNABLE MASK TOKEN: Used to replace masked positions during generation
        # ═══════════════════════════════════════════════════════════════════════
        # Shape: [1, 1, teacher_dims]
        # These tokens are randomly initialized and learned during training
        # They act as "placeholders" for missing information
        self.mask_token = nn.Parameter(torch.zeros(1, 1, teacher_dims))

        # ═══════════════════════════════════════════════════════════════════════
        # GENERATION BLOCK: Convolutional layers to reconstruct teacher features
        # ═══════════════════════════════════════════════════════════════════════
        # This block takes partially masked features and tries to regenerate
        # the complete teacher feature map
        # Architecture: Conv → ReLU → Conv
        self.generation = nn.Sequential(
                nn.Conv2d(teacher_dims, teacher_dims, kernel_size=3, padding=1),
                nn.ReLU(inplace=True), 
                nn.Conv2d(teacher_dims, teacher_dims, kernel_size=3, padding=1))

    def forward(self,
                preds_S,    # Student's features: [low_s, high_s, mid_token]
                preds_T):   # Teacher's features: [low_t, high_t]
        """
        Compute the ViTKD loss by combining mimicking and generation losses.
        
        Args:
            preds_S (List): Student's feature maps
                - preds_S[0]: Shallow features from layers 0-1, shape [B, 2, N, D_student]
                - preds_S[1]: Deep features from last layer, shape [B, N, D_student]
                - preds_S[2]: (Optional) Mid-layer token for self-distillation
            
            preds_T (List): Teacher's feature maps
                - preds_T[0]: Shallow features from layers 0-1, shape [B, 2, N, D_teacher]
                - preds_T[1]: Deep features from last layer, shape [B, N, D_teacher]
        
        Where:
            B = Batch size (e.g., 32)
            N = Number of patch tokens (e.g., 196 for 14×14 grid)
            D = Embedding dimension (192 for student, 384 for teacher)
        
        Returns:
            torch.Tensor: Combined loss (loss_lr + loss_gen)
        """
        
        # ═══════════════════════════════════════════════════════════════════════
        # EXTRACT FEATURES
        # ═══════════════════════════════════════════════════════════════════════
        
        # Shallow layer features (first 2 transformer layers)
        low_s = preds_S[0]  # Student: [B, 2, N, D_student] - captures early attention patterns
        low_t = preds_T[0]  # Teacher: [B, 2, N, D_teacher]
        
        # Deep layer features (last transformer layer)
        high_s = preds_S[1]  # Student: [B, N, D_student] - captures semantic information
        high_t = preds_T[1]  # Teacher: [B, N, D_teacher]

        # Get batch size for loss normalization
        B = low_s.shape[0]
        
        # Define MSE loss with sum reduction (will normalize by batch size manually)
        loss_mse = nn.MSELoss(reduction='sum')

        # ═══════════════════════════════════════════════════════════════════════
        # PART 1: MIMICKING MODULE (Shallow Layers Distillation)
        # ═══════════════════════════════════════════════════════════════════════
        # Goal: Make student's early layers focus on similar patterns as teacher
        # Method: Direct alignment + MSE loss
        
        if self.align2 is not None:
            # Need to align dimensions: student (192) → teacher (384)
            
            # Process each shallow layer separately
            for i in range(2):  # Iterate over layer 0 and layer 1
                if i == 0:
                    # Transform layer 0: [B, N, 192] → [B, N, 384]
                    # Then add dimension: [B, N, 384] → [B, 1, N, 384]
                    xc = self.align2[i](low_s[:,i]).unsqueeze(1)
                else:
                    # Transform layer 1: [B, N, 192] → [B, N, 384]
                    # Add dimension: [B, N, 384] → [B, 1, N, 384]
                    # Concatenate with layer 0: [B, 1, N, 384] + [B, 1, N, 384] → [B, 2, N, 384]
                    xc = torch.cat((xc, self.align2[i](low_s[:,i]).unsqueeze(1)), dim=1)
        else:
            # Dimensions already match, no alignment needed
            xc = low_s

        # Compute mimicking loss: MSE between aligned student and teacher shallow features
        # Normalize by batch size and scale by alpha
        # Formula: L_lr = (1/B) * α * MSE(xc, low_t)
        loss_lr = loss_mse(xc, low_t) / B * self.alpha_vitkd

        # ═══════════════════════════════════════════════════════════════════════
        # PART 2: GENERATION MODULE (Deep Layer Distillation)
        # ═══════════════════════════════════════════════════════════════════════
        # Goal: Force student to generate teacher's features from partial information
        # Method: Random masking + learnable mask tokens + convolutional generation
        
        # Step 1: Align student's deep features to teacher's dimension
        if self.align is not None:
            # Transform: [B, N, 192] → [B, N, 384]
            x = self.align(high_s)
        else:
            # No alignment needed
            x = high_s

        # Step 2: Apply random masking
        # Get feature dimensions
        B, N, D = x.shape  # B=batch, N=196 tokens, D=384 dimensions
        
        # Randomly mask λ% of tokens (default: 50%)
        # Returns:
        #   x: Only the unmasked tokens [B, N*(1-λ), D]
        #   mat: Binary mask [B, N] where 1=masked, 0=visible
        #   ids: Shuffle indices to restore original order
        #   ids_masked: Indices of masked positions
        x, mat, ids, ids_masked = self.random_masking(x, self.lambda_vitkd)
        
        # Step 3: Insert learnable mask tokens at masked positions
        # Create mask tokens: [1, 1, D] → [B, N_masked, D]
        # N_masked = N - x.shape[1] (number of masked tokens)
        mask_tokens = self.mask_token.repeat(B, N - x.shape[1], 1)
        
        # Concatenate unmasked tokens with mask tokens
        # [B, N_visible, D] + [B, N_masked, D] → [B, N, D]
        x_ = torch.cat([x, mask_tokens], dim=1)
        
        # Unshuffle to restore original token order
        # Now masked positions have learnable mask tokens
        x = torch.gather(x_, dim=1, index=ids.unsqueeze(-1).repeat(1, 1, D))
        
        # Convert mask to correct shape for element-wise multiplication later
        mask = mat.unsqueeze(-1)  # [B, N] → [B, N, 1]

        # Step 4: Apply generative block (convolutional layers)
        # Reshape from sequence to 2D grid: [B, N, D] → [B, D, H, W]
        hw = int(N**0.5)  # Calculate grid size (196 → 14)
        x = x.reshape(B, hw, hw, D).permute(0, 3, 1, 2)  # [B, 14, 14, 384] → [B, 384, 14, 14]
        
        # Pass through convolutional layers: [B, 384, 14, 14] → [B, 384, 14, 14]
        # This allows the model to use spatial context to fill in masked regions
        x = self.generation(x)
        
        # Reshape back to sequence: [B, 384, 14, 14] → [B, N, D]
        x = x.flatten(2).transpose(1, 2)

        # Step 5: Compute generation loss (only on masked tokens)
        # Multiply by mask to focus loss only on reconstructed (masked) positions
        # Element-wise multiplication: [B, N, D] * [B, N, 1] → [B, N, D]
        loss_gen = loss_mse(torch.mul(x, mask), torch.mul(high_t, mask))
        
        # Normalize by batch size, scale by beta, and divide by lambda
        # Division by lambda compensates for having fewer masked tokens to compare
        # Formula: L_gen = (1/B) * (β/λ) * MSE(x ⊙ mask, high_t ⊙ mask)
        loss_gen = loss_gen / B * self.beta_vitkd / self.lambda_vitkd
        
        # ═══════════════════════════════════════════════════════════════════════
        # RETURN COMBINED LOSS
        # ═══════════════════════════════════════════════════════════════════════
        # Total ViTKD loss = Shallow mimicking loss + Deep generation loss
        return loss_lr + loss_gen

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by shuffling tokens based on random noise.
        
        This function is inspired by Masked Autoencoders (MAE). It randomly selects
        which tokens to keep and which to mask, simulating partial information.
        
        Args:
            x (torch.Tensor): Input features, shape [N, L, D]
                N = batch size
                L = sequence length (number of tokens)
                D = embedding dimension
            
            mask_ratio (float): Ratio of tokens to mask (e.g., 0.5 = mask 50%)
        
        Returns:
            x_keep (torch.Tensor): Unmasked tokens, shape [N, L*(1-mask_ratio), D]
            mask (torch.Tensor): Binary mask, shape [N, L], where 1=masked, 0=visible
            ids_restore (torch.Tensor): Indices to unshuffle tokens back to original order
            ids_masked (torch.Tensor): Indices of the masked tokens
        
        Example:
            Input: x with L=196 tokens, mask_ratio=0.5
            Output: x_keep with 98 tokens, mask indicating which 98 were masked
        """
        N, L, D = x.shape  # batch, length, dimension
        
        # Calculate how many tokens to keep (visible)
        len_keep = int(L * (1 - mask_ratio))  # e.g., 196 * 0.5 = 98 tokens to keep
        
        # Generate random noise for each token in each sample
        # Shape: [N, L], values in [0, 1]
        noise = torch.rand(N, L, device=x.device)
        
        # Sort tokens by noise values (ascending order)
        # Tokens with smaller noise values will be kept, larger values will be masked
        # ids_shuffle contains the sorting indices
        ids_shuffle = torch.argsort(noise, dim=1)  # [N, L]
        
        # Get inverse permutation to restore original order later
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # [N, L]

        # Keep the first subset (tokens with smallest noise values)
        ids_keep = ids_shuffle[:, :len_keep]  # [N, len_keep] - indices of visible tokens
        ids_masked = ids_shuffle[:, len_keep:L]  # [N, L-len_keep] - indices of masked tokens

        # Gather the visible tokens using their indices
        # Expand ids_keep to match feature dimension: [N, len_keep] → [N, len_keep, D]
        # Use gather to select only the visible tokens
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate binary mask: 0 = keep, 1 = remove
        mask = torch.ones([N, L], device=x.device)  # Start with all 1s (all masked)
        mask[:, :len_keep] = 0  # Set first len_keep positions to 0 (visible)
        
        # Unshuffle mask to get binary indicator for original token positions
        # This maps the mask back to the original token order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_keep, mask, ids_restore, ids_masked