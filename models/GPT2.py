from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.CUSTOM_INIT = 1 # Flag for special init

        self.register_buffer('mask', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape # batch, sequence length, embedding size
        # nh = number of heads, hs = head size, and embeddings = nh * hs
        qkv = self.c_attn(x) # (B, T, C) -> (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embd, dim=2) # (B, T, C) -> (B, T, C) * 3

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, C) -> (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, C) -> (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, C) -> (B, nh, T, hs) 

        # attention
        head_size = k.size(-1)
        wei = q @ k.transpose(-2, -1) * head_size ** (-0.5)
        wei = wei.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, nh, T, T)

        # weighted average of values
        out = wei @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C) # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C=nh*hs)

        out = self.c_proj(out)
        return out



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.CUSTOM_INIT = 1 # Flag for special init
        self.gelu = nn.GELU(approximate='tanh') 
        # GPT 2 uses tanh approximation of GELU; nowadays, the exact version is used

    def forward(self, x):
        x = self.gelu(self.c_fc(x)) 
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # Residual connection
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class Hyperparameters:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd), # Weight of token embeddings
                wpe = nn.Embedding(config.block_size, config.n_embd), # Weight of positional embeddings
                h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f = nn.LayerNorm(config.n_embd)
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Sharing Scheme
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'CUSTOM_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # 2 layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)



    def forward(self, idx, targets=None):
        # idx: (B, T)
        device = idx.device
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequence length {T} is greater than block size {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(torch.arange(0, T, dtype=torch.long, device=device))
        x = tok_emb + pos_emb

        # Forwarding in blocks
        # print("blocks")
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss


    @classmethod
    def from_pretrained(cls, model_name: str):
        # The model name must be one of the following:
        assert model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768), # 124 M Parameters
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350 M Parameters
            'gpt2-large': dict(n_layers=36, n_head=20, n_embd=1280), # 774 M Parameters
            'gpt2-xl': dict(n_layers=48, n_head=25, n_embd=1600) # 1.558 B Parameters
        }[model_name]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        # Now that we have all the params, we load it into dataclass
        config = Hyperparameters(**config_args)
        model = GPT(config)
        sd = model.state_dict()


        # Ignore the ones ending with attn.mask
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.mask')]


        # for k in sd_keys:
        #     print(k)
        # Load the Pretrained model
        from transformers import GPT2LMHeadModel
        model_pre= GPT2LMHeadModel.from_pretrained(model_name)
        sd_pre = model_pre.state_dict()
        # print(sd_pre.keys())
        sd_keys_pre = sd_pre.keys()
        sd_keys_pre = [k for k in sd_keys_pre if not k.endswith('.attn.bias')]
        sd_keys_pre = [k for k in sd_keys_pre if not k.endswith('.attn.masked_bias')]

        # Manually checking dimensions, some keys are transposed for whatever reason
        transposed_keys = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        assert len(sd_keys) == len(sd_keys_pre), f"{len(sd_keys)} != {len(sd_keys_pre)}"
        for k in sd_keys_pre:
            if any(k.endswith(tk) for tk in transposed_keys):
                assert sd_pre[k].T.shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_pre[k].T)
            else:
                # vanilla copy
                assert sd_pre[k].shape == sd[k].shape, f"Key: {k}, {sd_pre[k].shape} != {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_pre[k])

        return model



if __name__ == '__main__':
    model = GPT.from_pretrained('gpt2')
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'

    print(f"Using {device}")


    model.eval()
    model.to(device)
    # Setup
    num_return_sequences = 5
    max_length = 30
    # prompt = "Cristiano Ronaldo is one of the greatest players of all time. "
    prompt = "Hello, I'm a language model,"

    # Prefix Tokens
    import tiktoken
    encoder = tiktoken.get_encoding('gpt2')
    prefix_tokens = encoder.encode(prompt)
    prefix_tokens = torch.tensor(prefix_tokens, dtype=torch.long).unsqueeze(0)
    prefix_tokens = prefix_tokens.repeat(num_return_sequences, 1)

    x = prefix_tokens.to(device) # (B=num_return_sequences, T=prefix_length)
    print(x)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x) # (B, T, vocab_size)

            # Take logits at last position, because that's the next token
            logits = logits[:, -1, :] # (B, vocab_size)

            # Get the probabilities
            probs = F.softmax(logits, dim=-1)

            # Use topk
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

            # Sample from topk
            ix = torch.multinomial(topk_probs, num_samples=1) # (B, 1)

            xcol = torch.gather(topk_indices, 1, ix)

            x = torch.cat((x, xcol), dim=1)


    # Decode the tokens
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = encoder.decode(tokens)
        print("---", decoded)







    

    



