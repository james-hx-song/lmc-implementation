import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
block_size = 8
batch_size = 32
max_iters = 5000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iter = 200
n_embed = 32
n_head = 4
n_layers = 4
dropout = 0.1

print(f"Using {device}")

# ----------------- Data Preprocessing ----------------- #
torch.manual_seed(1337)

with open("input.txt", 'r') as f:
  text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda i: ''.join([itos[e] for e in i])
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size, )) # Random Batches
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x = x.to(device)
  y = y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'eval']:
    # Average losses over multiple batches
    losses = torch.zeros(eval_iter)
    for i in range(eval_iter):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[i] = loss.item()
    out[split] = losses.mean()
  model.train()

  return out




torch.manual_seed(1337)

# ----------------- Model ----------------- #

class Head(nn.Module):
  def __init__(self, head_size):
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

# Implementing Multi-head Attention
class MultiHeadAttention(nn.Module):
  """ Multiple Heads of self-attention in parallel"""

  def __init__(self, num_heads, head_size) -> None:
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    # Projection
    self.proj = nn.Linear(n_embed, n_embed)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

# Implementing Feed-Forward Network
class FFN(nn.Module):
  def __init__(self, n_embed) -> None:
    super().__init__()
    self.layer = nn.Sequential(
      nn.Linear(n_embed, 4*n_embed),
      nn.ReLU(),
      nn.Linear(4*n_embed, n_embed), # Projection Layer
      nn.Dropout(dropout),
    )
  def forward(self, x):
    return self.layer(x)

# Blocks of MultiHeadAttention and FFN
class Block(nn.Module):
  ''' Transformer Block'''

  def __init__(self, n_embd, n_head):
    # n_embd is the embedding dimension, n_head is the number of heads we'd like
    super().__init__()
    self.sa = MultiHeadAttention(n_head, n_embd//n_head) # Communication
    self.ffwd = FFN(n_embd) # Computation
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    # return self.ffwd(self.sa(x))
    # residual connection
    x = x + self.sa(self.ln1(x)) # Layer Normalization
    x = x + self.ffwd(self.ln2(x))
    return x

class BigramLanguageModel(nn.Module):

  def __init__(self) -> None:
    super().__init__()

    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)

    # positional embedding
    self.position_embedding_table = nn.Embedding(block_size, n_embed) # each position of 0 to block_size-1 will also be embedded
    self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layers)])
    # self.lm_head = nn.Linear(n_embed, vocab_size)
    # # self.sa_head = Head(n_embed) # Self Attention Head
    # self.sa_heads = MultiHeadAttention(4, n_embed//4)
    # # FFN
    # self.ffn = FFN(n_embed)
    self.lm_head = nn.Linear(n_embed, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    token_embeddings = self.token_embedding_table(idx) # (B, T, C=n_embed)
    position_embeddings = self.position_embedding_table(torch.arange(T, device=device)) # (T, C=n_embed)
    x = token_embeddings + position_embeddings # (B, T, C=n_embed)
    # x = self.sa_heads(x)
    # x = self.ffn(x)
    x = self.blocks(x)
    logits = self.lm_head(x)
    if targets is None:
      loss = None
    else:
       # Note that this is Batch X Time X Channels
      B, T, C = logits.shape
      # Negative Log Likelihood Loss
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets) # Entropy wants it to be batch X time X channels
    return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
      # Crop idx to the last block_size tokens
      idx_cond = idx[:, -block_size:]
      # get the predictions
      logits, loss = self(idx_cond)

      logits = logits[:, -1, :]

      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx


# --------------- Training ---------------- 

model = BigramLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

batch_size = 32
for steps in range(max_iters):
  xb, yb = get_batch('train')

  if steps % 1000 == 0:
    out = estimate_loss()
    print(f"Step {steps}: Train Loss: {out['train']}, Eval Loss: {out['eval']}")

  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True)

  loss.backward()
  optimizer.step()
out = estimate_loss()
print(f"Final: Train Loss: {out['train']}, Eval Loss: {out['eval']}")
print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

