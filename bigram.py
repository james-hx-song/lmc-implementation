import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
block_size = 8
batch_size = 32
max_iters = 10000
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iter = 200
n_embed = 32


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

class BigramLanguageModel(nn.Module):

  def __init__(self) -> None:
    super().__init__()

    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)

    # positional embedding
    self.position_embedding_table = nn.Embedding(block_size, n_embed) # each position of 0 to block_size-1 will also be embedded
    self.lm_head = nn.Linear(n_embed, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    token_embeddings = self.token_embedding_table(idx) # (B, T, C=n_embed)
    position_embeddings = self.position_embedding_table(torch.arange(T, device=device)) # (T, C=n_embed)
    x = token_embeddings + position_embeddings # (B, T, C=n_embed)
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
      # get the predictions
      logits, loss = self(idx)

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
for steps in range(10000):
  xb, yb = get_batch('train')

  if steps % 1000 == 0:
    out = estimate_loss()
    print(f"Step {steps}: Train Loss: {out['train']}, Eval Loss: {out['eval']}")

  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True)

  loss.backward()
  optimizer.step()
print(loss.item())

