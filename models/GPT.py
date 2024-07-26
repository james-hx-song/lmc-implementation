from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

# ----------------- Model Architecture ----------------- #
# We follow the naming convention of GPT-2 (GPT2LMHeadModel) from HuggingFace

# Default is 124 M GPT 2

@dataclass
class GPTConfig:
    vocab_size: int = 50257 # (Radford et al. 2020)
    n_embed: int = 768
    block_size: int = 1024
    batch_size: int = 512
    n_layer: int = 12
    n_head: int = 12

@dataclass
class MinGPTConfig:
    vocab_size: int = 50257 # (Radford et al. 2020)
    n_embed: int = 768
    block_size: int = 256
    batch_size: int = 16
    n_layer: int = 5
    n_head: int = 5

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

        out = self.c_proj(attn_vec) # (B, T, C)

        return out

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, config.n_embed*4) # Both Vaswani eta al. and GPT-2 use 4x
        self.c_proj = nn.Linear(config.n_embed*4, config.n_embed)

        # GPT-2 uses GELU; Vaswani et al. uses ReLU
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.c_fc(x))
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.attn = Attention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        # Vaswani et al. use LayerNorm(x + Attention(LayerNorm(x)))
        x = x + self.attn(self.ln_1(x)) # Residual Connection
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embed), #  Token Embeddings
                wpe = nn.Embedding(config.block_size, config.n_embed), #  Positional Embeddings
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # Transformer Blocks
                ln_f = nn.LayerNorm(config.n_embed) # Layer Norm
            )
        )

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        # Weight Sharing Scheme (Vaswani et al. 2017):
        # "In our model, we share the same weight matrix between the two embedding layers and the pre-softmax
        # linear transformation, similar to [30]". 
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            if module == self.transformer.wte:
                torch.nn.init.normal_(module.weight, std=0.02)
            elif module == self.transformer.wpe:
                torch.nn.init.normal_(module.weight, std=0.01)
            

    def forward(self, x, y=None):
        # Token + Position
        x = self.transformer.wte(x) + self.transformer.wpe(torch.arange(x.size(1), device=x.device))

        # Transformer Blocks
        for block in self.transformer.h:
            x = block(x)
        
        # According to GPT 2 repo, they add layer norm at the end
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return logits, loss
    
    def generate(self, prefix_tokens, max_len, num_copies, device):
        prefix_tokens = torch.tensor(prefix_tokens, dtype=torch.long).unsqueeze(0)
        prefix_tokens = prefix_tokens.repeat(num_copies, 1)

        x = prefix_tokens.to(device)
        while x.size(1) < max_len:
            with torch.no_grad():
                # print(x.shape)
                logits, _ = self(x)
                # print(logits.shape)

                logits = logits[:, -1, :] # (B, vocab_size) gets last token
                
                proba = F.softmax(logits, dim=-1)

                # Sample from the distribution
                next_idx = torch.multinomial(proba, num_samples=1)

                x = torch.cat((x, next_idx), dim=1)

        return x.tolist()

if __name__ == '__main__':
    prompt = "Hello, I am a Large Language Model. "
    num_copies = 2
    max_len = 1000
    device = 'mps'

    model = GPT(GPTConfig())
    model.to(device)

    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(prompt)
    
    next_tokens = model.generate(tokens, max_len, num_copies, device)

    for i in range(num_copies):
        tokens = next_tokens[i, :max_len]
        text = enc.decode(tokens)
        print(text)
    


    
    





    # x = torch.randint(0, 50257, (2, 1024))
    # y = torch.randint(0, 50257, (2, 1024))
    # # print(x.shape, y.shape)
    # logits, loss = model(x, y)
    # print(logits.shape, loss.item())