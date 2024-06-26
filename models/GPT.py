import torch
import torch.nn as nn
from torch.nn import functional as F



class Head(nn.Module):
  def __init__(self, head_size, n_embed, block_size, dropout):
    super().__init__()

    self.key = nn.Linear(n_embed, head_size, bias=False)
    self.query = nn.Linear(n_embed, head_size, bias=False)
    self.value = nn.Linear(n_embed, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)
    q = self.query(x)

    # Computing the attention scores
    wei = q @ k.transpose(-2, -1) * C**(-0.5) # C is the headsize # (B, T, C) @ (B, C, T) -> (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
    wei = F.softmax(wei, dim=-1) # (B, T, T)
    wei = self.dropout(wei)

    # Perform the weighted aggregation of the values
    v = self.value(x) # (B, T, C)
    out = wei @ v # (B,T, T) @ (B, T, C) -> (B, T, C)

    return out
  
class MultiHeadAttention(nn.Module):
  """ Multiple Heads of self-attention in parallel"""

  def __init__(self, num_heads, head_size, n_embed, block_size, dropout) -> None:
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size, n_embed, block_size, dropout) for _ in range(num_heads)])
    # Projection
    self.proj = nn.Linear(n_embed, n_embed)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))
  

class FFN(nn.Module):
  def __init__(self, n_embed, dropout) -> None:
    super().__init__()
    self.layer = nn.Sequential(
      nn.Linear(n_embed, 4*n_embed),
      nn.ReLU(),
      nn.Linear(4*n_embed, n_embed), # Projection Layer
      nn.Dropout(dropout),
    )
  def forward(self, x):
    return self.layer(x)
  

class Block(nn.Module):
  ''' Transformer Block'''

  def __init__(self, n_embd, n_head, block_size, dropout):
    # n_embd is the embedding dimension, n_head is the number of heads we'd like
    super().__init__()
    self.sa = MultiHeadAttention(n_head, n_embd//n_head, n_embd, block_size, dropout) # Communication
    self.ffwd = FFN(n_embd, dropout) # Computation
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    # return self.ffwd(self.sa(x))
    # residual connection
    x = x + self.sa(self.ln1(x)) # Layer Normalization
    x = x + self.ffwd(self.ln2(x))
    return x
  

class BigramLanguageModel(nn.Module):

  def __init__(self, vocab_size, **kwargs) -> None:
    super().__init__()
    self.vocab_size = vocab_size
    self.n_embed = kwargs.get('n_embed', 32)
    self.block_size = kwargs.get('block_size', 8)
    self.n_head = kwargs.get('n_head', 4)
    self.n_layers = kwargs.get('n_layers', 4)
    self.dropout = kwargs.get('dropout', 0.1)
    self.device = kwargs.get('device', 'cpu')

    self.token_embedding_table = nn.Embedding(vocab_size, self.n_embed)
    # positional embedding
    self.position_embedding_table = nn.Embedding(self.block_size, self.n_embed) # each position of 0 to block_size-1 will also be embedded
    self.blocks = nn.Sequential(*[Block(self.n_embed, n_head=self.n_head, block_size=self.block_size, dropout=self.dropout) for _ in range(self.n_layers)])
    # self.lm_head = nn.Linear(n_embed, vocab_size)
    # # self.sa_head = Head(n_embed) # Self Attention Head
    # self.sa_heads = MultiHeadAttention(4, n_embed//4)
    # # FFN
    # self.ffn = FFN(n_embed)
    self.lm_head = nn.Linear(self.n_embed, vocab_size)

  def forward(self, idx, target=None):
    B, T = idx.shape
    token_embeddings = self.token_embedding_table(idx) # (B, T, C=n_embed)
    position_embeddings = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C=n_embed)
    x = token_embeddings + position_embeddings # (B, T, C=n_embed)
    # x = self.sa_heads(x)
    # x = self.ffn(x)
    x = self.blocks(x)
    logits = self.lm_head(x)
    if target is None:
      loss = None
    else:
       # Note that this is Batch X Time X Channels
      B, T, C = logits.shape
      # Negative Log Likelihood Loss
      pred = logits.view(B*T, C)
      target = target.view(B*T)
      loss = F.cross_entropy(pred, target) # Entropy wants it to be batch X time X channels
    return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
      # Crop idx to the last block_size tokens
      idx_cond = idx[:, -self.block_size:]
      # get the predictions
      logits, loss = self(idx_cond)

      logits = logits[:, -1, :]

      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

if __name__ == "__main__":
  with open("input.txt", 'r') as f:
    text = f.read()
  vocab = list(set(text))
  print(f"Bigram Language Model has {sum(p.numel() for p in BigramLanguageModel(len(vocab)).parameters() if p.requires_grad)} parameters.")
