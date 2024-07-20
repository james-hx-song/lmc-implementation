from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- Model Architecture ----------------- #
# We follow the naming convention of GPT-2 (GPT2LMHeadModel) from HuggingFace

# Following the GPT2 Repo (124 M)
@dataclass
class GPT2Config:
    vocab_size: int = 50257 # (Radford et al.)
    n_embed: int = 768
    block_size: int = 1024
    batch_size: int = 512
    n_layer: int = 12
    n_head: int = 12

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embed, config.n_embed*3)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        self.n_head = config.n_head

        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x) # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=2) # (B, T, C) x 3

        # Now we want to split the heads, so # (B, nh, T, hs = C/nh)
        q = q.view(B, T, self.n_head, C// self.n_head).transpose(1, 2) # (B, T, C) -> (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V
        head_size = q.size(-1)
        attn_score = (q @ k.transpose(-2, -1)) * (head_size ** (-0.5)) # (B, nh, T, T)
        # Decoder-only transformer, so mask the future tokens
        attn_score = attn_score.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn_prob = F.softmax(attn_score, dim=-1)

        # Apply the attention to the values
        attn_vec = attn_prob @ v # (B, nh, T, hs)
        # Concatenate the heads
        attn_vec = attn_vec.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)

        out = self.c_proj(attn_vec) # (B, T, C

        return out

        


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.attn = Attention(config)


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embed), #  Token Embeddings
                wpe = nn.Embedding(config.block_size, config.n_embed), #  Positional Embeddings
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]) # Transformer Blocks
            )
        )

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)



        

if __name__ == '__main__':
    model = GPT(GPT2Config())
    
    for k, v in model.state_dict().items():
        print(k, v.shape)