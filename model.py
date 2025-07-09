import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x, context):
        h = self.heads
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class BiDirectionalFusionBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm_self1 = nn.LayerNorm(dim)
        self.self_attn1 = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_self2 = nn.LayerNorm(dim)
        self.self_attn2 = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_cross1 = nn.LayerNorm(dim)
        self.cross_attn1 = CrossAttention(dim, heads, dim_head, dropout)
        self.norm_cross2 = nn.LayerNorm(dim)
        self.cross_attn2 = CrossAttention(dim, heads, dim_head, dropout)
        self.norm_ff1 = nn.LayerNorm(dim)
        self.ff1 = FeedForward(dim, mlp_dim, dropout)
        self.norm_ff2 = nn.LayerNorm(dim)
        self.ff2 = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x1, x2):
        x1 = x1 + self.self_attn1(self.norm_self1(x1), self.norm_self1(x1), self.norm_self1(x1), need_weights=False)[0]
        x2 = x2 + self.self_attn2(self.norm_self2(x2), self.norm_self2(x2), self.norm_self2(x2), need_weights=False)[0]
        x1 = x1 + self.cross_attn1(self.norm_cross1(x1), self.norm_cross1(x2))
        x2 = x2 + self.cross_attn2(self.norm_cross2(x2), self.norm_cross2(x1))
        x1 = x1 + self.ff1(self.norm_ff1(x1))
        x2 = x2 + self.ff2(self.norm_ff2(x2))
        return x1, x2

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, img_size=224, dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
    
    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        return x

class VisionTransformerWithGradCAM(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000, dim=768, depth=6, 
                 heads=12, mlp_dim=3072, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(3, patch_size, image_size, dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.layers = nn.ModuleList([
            BiDirectionalFusionBlock(dim, heads, dim_head, mlp_dim, dropout) 
            for _ in range(depth)
        ])
        self.cls_head = nn.Sequential(
            nn.LayerNorm(dim * 2 + 2),  # +2 for CAM scores
            nn.Linear(dim * 2 + 2, num_classes)
        )
        self.cam_conv1 = nn.Conv1d(dim, 1, kernel_size=1)
        self.cam_conv2 = nn.Conv1d(dim, 1, kernel_size=1)
        
        # For Grad-CAM
        self.last_x1 = None
        self.last_x2 = None
        self.gradients_x1 = None
        self.gradients_x2 = None
    
    def save_activation_and_grad(self, x1, x2):
        # Only save activations and register hooks during training
        if self.training:
            # Store detached copies of the activations
            self.last_x1 = x1.detach()
            self.last_x2 = x2.detach()
            
            # Only register hooks if the tensors require gradients
            if x1.requires_grad:
                x1.register_hook(lambda grad: setattr(self, 'gradients_x1', grad))
            if x2.requires_grad:
                x2.register_hook(lambda grad: setattr(self, 'gradients_x2', grad))
    
    def forward(self, img1, img2):
        # Ensure input tensors require gradients if in training mode
        if self.training:
            img1 = img1.requires_grad_(True)
            img2 = img2.requires_grad_(True)
        
        # Extract patch embeddings
        x1 = self.patch_embed(img1)
        x2 = self.patch_embed(img2)
        
        # Apply dropout
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        
        # Pass through bidirectional fusion blocks
        for layer in self.layers:
            x1, x2 = layer(x1, x2)
        
        # Save activations and register hooks for Grad-CAM (only in training)
        if self.training:
            self.save_activation_and_grad(x1, x2)
        
        # Get class tokens
        cls1, cls2 = x1[:, 0], x2[:, 0]
        
        # Compute CAM scores
        cam_score1 = self.cam_conv1(x1.transpose(1, 2)).mean(dim=2)
        cam_score2 = self.cam_conv2(x2.transpose(1, 2)).mean(dim=2)
        
        # Concatenate class tokens and CAM scores
        out = torch.cat([cls1, cls2, cam_score1, cam_score2], dim=1)
        
        # Final classification
        return self.cls_head(out)
    
    def get_attention_maps(self, img1, img2):
        """Extract attention maps for visualization"""
        x1 = self.patch_embed(img1)
        x2 = self.patch_embed(img2)
        
        attention_maps = []
        for layer in self.layers:
            # Get attention weights from self-attention
            _, attn_weights1 = layer.self_attn1(
                layer.norm_self1(x1), 
                layer.norm_self1(x1), 
                layer.norm_self1(x1),
                need_weights=True
            )
            _, attn_weights2 = layer.self_attn2(
                layer.norm_self2(x2),
                layer.norm_self2(x2),
                layer.norm_self2(x2),
                need_weights=True
            )
            attention_maps.append((attn_weights1, attn_weights2))
            
            # Update features for next layer
            x1, x2 = layer(x1, x2)
            
        return attention_maps

def compute_gradcam(activations, gradients, target_size=(224, 224)):
    """
    Compute Grad-CAM heatmap
    Args:
        activations: Tensor of shape [batch_size, num_patches+1, dim]
        gradients: Tensor of shape [batch_size, num_patches+1, dim]
        target_size: Size to resize the heatmap to
    """
    # Remove class token
    activations = activations[:, 1:]
    gradients = gradients[:, 1:]
    
    # Compute weights (global average pooling of gradients)
    weights = torch.mean(gradients, dim=1, keepdim=True)  # [batch_size, 1, dim]
    
    # Weighted combination of activation maps
    cam = torch.bmm(weights, activations.transpose(1, 2))  # [batch_size, 1, num_patches]
    cam = F.relu(cam)
    
    # Reshape to 2D and normalize
    num_patches = int(cam.shape[-1] ** 0.5)
    cam = cam.view(cam.size(0), 1, num_patches, num_patches)
    
    # Resize to target size
    cam = F.interpolate(cam, size=target_size, mode='bilinear', align_corners=False)
    
    # Normalize
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    
    return cam.squeeze(1)  # [batch_size, H, W]
