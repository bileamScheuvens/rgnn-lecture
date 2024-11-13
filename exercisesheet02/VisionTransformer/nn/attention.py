import torch
import torch.nn as nn
import math
from einops import rearrange
from einops.layers.torch import Rearrange

class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_size, num_heads):
        """
        Initializes the attention layer.

        Args:
            embed_size (int): The embedding size of the input.
            num_heads (int): The number of attention heads.
        """
        super(SelfAttentionLayer, self).__init__()
        self.embed_size = embed_size  # The embedding dimension of the model
        self.num_heads = num_heads    # Number of attention heads
        
        # Ensure the embedding size is divisible by the number of heads
        assert embed_size % num_heads == 0, "Embedding size must be divisible by num_heads"
        
        # The dimension of each attention head
        self.head_dim = embed_size // num_heads

        # Define linear layers for queries, keys, and values
        self.q_linear = nn.Linear(embed_size, embed_size)  # Linear transformation for queries
        self.k_linear = nn.Linear(embed_size, embed_size)  # Linear transformation for keys
        self.v_linear = nn.Linear(embed_size, embed_size)  # Linear transformation for values
        
        # Final linear layer after concatenating attention heads
        self.output_linear = nn.Linear(embed_size, embed_size)
    
    def forward(self, x):
        """
        Performs the forward pass of the attention layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, embed_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, embed_size).
        """
        batch_size, seq_length, embed_size = x.size()
        
        # Linear projections of queries, keys, and values
        queries = self.q_linear(x)  # Shape: (batch_size, seq_length, embed_size)
        keys    = self.k_linear(x)  # Shape: (batch_size, seq_length, embed_size)
        values  = self.v_linear(x)  # Shape: (batch_size, seq_length, embed_size)

        # Split the embedding dimension into multiple heads and rearrange the tensor
        # Reshape and transpose to get shape: (batch_size, num_heads, seq_length, head_dim)
        queries = queries.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(-2, -3)
        keys = keys.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(-2, -3)
        values = values.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(-2, -3)

        # Compute scaled dot-product attention
        # Compute attention scores by taking the dot product between queries and keys
        # and scaling by the square root of the head dimension
        # scores shape: (batch_size, num_heads, seq_length, seq_length)
        scores = (queries @ keys.transpose(-2,-1)) / math.sqrt(self.head_dim)

        # Compute the attention weights using the softmax function
        weights = torch.nn.functional.softmax(scores, dim=-1)

        # Multiply attention weights with values to get the context vector
        # attention_weights: (batch_size, num_heads, seq_length, seq_length)
        # values: (batch_size, num_heads, seq_length, head_dim)
        # Output: (batch_size, num_heads, seq_length, head_dim)
        context = weights @ values

        # Concatenate the heads and pass through the final linear layer
        # First, transpose and reshape to combine the heads
        # Reshape from (batch_size, num_heads, seq_length, head_dim) to (batch_size, seq_length, embed_size)
        context = context.transpose(-3,-2).reshape(batch_size, seq_length, -1)

        # Apply the final linear transformation
        out = self.output_linear(context)  # Shape: (batch_size, seq_length, embed_size)

        return out

class FeedForwardLayer(nn.Module):
    def __init__(self, embed_size, expansion_factor=4):
        """
        Initializes the feed-forward layer.

        Args:
            embed_size (int): The embedding size of the input.
            expansion_factor (int): Factor to expand the hidden layer size in feed-forward network.
        """
        super(FeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(embed_size, embed_size * expansion_factor)  # First linear layer
        self.activation = nn.SiLU()  # Activation function
        self.fc2 = nn.Linear(embed_size * expansion_factor, embed_size)  # Second linear layer

    def forward(self, x):
        """
        Performs the forward pass of the feed-forward layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, embed_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, embed_size).
        """
        x = self.fc1(x)         # Apply the first linear transformation
        x = self.activation(x)  # Apply the activation function
        x = self.fc2(x)         # Apply the second linear transformation
        return x

class AttentionBlock(nn.Module):
    def __init__(self, embed_size, num_heads, expansion_factor=4):
        """
        Initializes the attention block, consisting of a self-attention layer and a feed-forward layer.

        Args:
            embed_size (int): The embedding size of the input.
            num_heads (int): The number of attention heads.
            expansion_factor (int): Factor to expand the hidden layer size in feed-forward network.
        """
        super(AttentionBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)  # Layer normalization before the self-attention layer
        self.self_attention = SelfAttentionLayer(embed_size, num_heads)  # Self-attention layer
        self.norm2 = nn.LayerNorm(embed_size)  # Layer normalization before the feed-forward layer
        self.feed_forward = FeedForwardLayer(embed_size, expansion_factor)  # Feed-forward layer

    def forward(self, x):
        """
        Performs the forward pass of the attention block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, embed_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, embed_size).
        """
        # Apply layer normalization before the self-attention layer (pre-norm)
        x_norm = self.norm1(x)

        # Pass through the self-attention layer
        attn_out = self.self_attention(x_norm)

        # Residual connection
        x = x + attn_out

        # Apply layer normalization before the feed-forward layer (pre-norm)
        x_norm = self.norm2(x)

        # Pass through the feed-forward layer
        ff_out = self.feed_forward(x_norm)

        # Residual connection
        x = x + ff_out

        return x

class PatchEmbedding(nn.Module):
    def __init__(self, num_frames, embed_size, patch_size, image_height, image_width, mask_percentage=0.5):
        """
        Initializes the patch embedding layer.

        Args:
            num_frames (int): Number of frames in the input video.
            embed_size (int): The embedding size of the transformer.
            patch_size (int): The size of each image patch (e.g., 16 for 16x16 patches).
        """
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.num_frames = num_frames

        # Compute number of patches per frame
        self.num_patches_per_frame = (image_height // patch_size) * (image_width // patch_size)

        # compute total number of patches
        num_patches = (image_height // patch_size) * (image_width // patch_size) * num_frames

        # Project patches into embedding dimension (treat each frame seperatly!)
        self.makepatch = torch.nn.Unfold(kernel_size=(patch_size, patch_size), stride=patch_size)
        self.patchproj = torch.nn.Linear(patch_size*patch_size, embed_size)
        
        #------------------------------ attempt at positional embedding with scales, didn't work right away
        # # Positional Embedding
        # scales = torch.arange(0, 10000, step=100).unsqueeze(1)
        # xrange = torch.arange(image_width).repeat(image_height).unsqueeze(0)
        # yrange = torch.arange(image_height).repeat(image_width).unsqueeze(0)
        # # shape: [scales, W*H]
        # xpos = torch.sin(2 * math.pi * scales * xrange)
        # ypos = torch.cos(2 * math.pi * scales * yrange)
        # # shape: [H*W]
        # self.xpos = xpos.sum(dim=0)
        # self.ypos = ypos.sum(dim=0)
        # self.pos_embed = torch.nn.Linear(image_width*image_height, embed_size)
        #------------------------------ old

        pe = torch.zeros(image_height * image_width, embed_size)
        position = torch.arange(0, image_height * image_width, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pos_embed = torch.nn.Parameter(pe.unsqueeze(0), requires_grad=False)

        # Mask percentage for masking patches
        self.mask_percentage = mask_percentage

    def mask_patches(self, x):
        """
        Masks a random subset of the patches in the input tensor.
        """
        
        # Get the batch size and number of patches
        batch_size, num_patches, embed_size = x.size()

        # ernumerate patches
        patches = torch.linspace(0, 1, num_patches, device=x.device)

        # reshape and repeat
        patches = patches.view(1, num_patches, 1).repeat(batch_size, 1, 1)

        # shuffle patch dimmension
        for b in range(batch_size):
            patches[b] = patches[b][torch.randperm(num_patches, device=x.device)]
        
        # Generate a random mask
        mask = (patches < self.mask_percentage).float()
        
        # Apply the mask to the input tensor
        x = x * mask

        return x, mask
    
    def patch_and_proj(self, x):
        batch_size, num_frames, height, width = x.shape
        x = x.reshape(-1, 1, height, width) # shape: (B * n_frames, 1, H, W)
        # print(f"batch_size {batch_size}, n_frames {num_frames}, HW {height} {width}")
        patched = self.makepatch(x) # shape: (B * n_frames, patch_size^2, n_patches)
        out = self.patchproj(patched.transpose(-1,-2)) # shape: (B *n_frames, n_patches, embed_dim)
        out = out.reshape(batch_size, -1, self.embed_size)# shape: (B, n_frames*n_patches, embed_dim)
        return out

    def forward(self, x):
        """
        Performs the forward pass of the patch embedding layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_frames, height, width).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_patches, embed_size).
        """
        if self.num_frames > 1:
            return self.predict(x)

        # x shape: (batch_size, num_frames, height, width)
        # After projection: (batch_size, num_patches, embed_size)
        x = self.patch_and_proj(x)

        # Mask a random number of patches
        x, mask = self.mask_patches(x)

        # x += self.pos_embed(self.xpos.to(x.device)) + self.pos_embed(self.ypos.to(x.device))
        x += x + self.pos_embed[:, :x.size(1), :]
        return x, mask


    def predict(self, x):
        """
        Performs the forward pass for prediction, masking out the last frame.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_frames, height, width).

        Returns:
            Tensor: Output tensor of shape (batch_size, total_patches, embed_size).
        """
        # Project the input
        x = self.patch_and_proj(x)
        # Calculate indices to mask out the last frame's patches
        total_patches = self.num_patches_per_frame * self.num_frames
        start_idx = total_patches - self.num_patches_per_frame

        # Mask out the last frame's patches
        mask = torch.ones(1, total_patches, 1, device=x.device)
        mask[:, start_idx:, :] = 0
        x = x * mask

        # Add position embeddings
        x += x + self.pos_embed[:, :x.size(1), :]
        return x, mask

class PrintShape(nn.Module):
    def __init__(self):
        super(PrintShape, self).__init__()
    def forward(self, x):
        print(x.shape)
        return x

class ReversePatchEmbedding(nn.Module):
    def __init__(self, embed_size, num_frames, patch_size, image_height, image_width):
        """
        Initializes the reverse patch embedding layer.

        Args:
            embed_size (int): The embedding size of the transformer.
            num_frames (int): Number of frames in the input video.
            patch_size (int): The size of each image patch (e.g., 16 for 16x16 patches).
            image_height (int): The height of the original image.
            image_width (int): The width of the original image.
        """
        super(ReversePatchEmbedding, self).__init__()
        self.embed_size = embed_size
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.num_pixels_per_patch = patch_size * patch_size
        self.image_height = image_height
        self.image_width = image_width

        self.proj = nn.Sequential(
            Rearrange('b (t h w) e -> (b t) e h w', h=image_height // patch_size, w=image_width // patch_size, t=num_frames),
            nn.ConvTranspose2d(embed_size, 1, kernel_size=patch_size, stride=patch_size),
            Rearrange('(b c) 1 h w -> b c h w', c=num_frames),
        )

    def forward(self, x):
        """
        Performs the forward pass of the reverse patch embedding layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_patches, embed_size).
            image_size (int): The size of the original image (assumed to be square).

        Returns:
            Tensor: Reconstructed image tensor of shape (batch_size, num_frames, height, width).
        """
        # Project embeddings back to flattened patches
        x = self.proj(x)  # Shape: (batch_size, num_patches, num_pixels_per_patch)
        return x