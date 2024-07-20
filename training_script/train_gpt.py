# GPT is written as a separate file because its training loop and configurations are significantly different from other models, 
# due to gradient accumulation and clipping grad norm
import torch
import time
import sys

# Setup / Config
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f"Using {device}")


# -----------------  Necessary Components: Models, Loaders, Optimizers ----------------- #
from models.GPT import GPT, GPTConfig
from scheduler.GPTScheduler import LRScheduler, LRConfig
from datasets.Langdata import LangDataLoader

model1 = GPT(GPTConfig()) 
sys.exit(0)
model1.to(device)

model2 = GPT(GPTConfig())
model2.to(device)

print(f"Model: {type(model1).__name__}")
print(f"with {sum(p.numel() for p in model1.parameters() if p.requires_grad)} parameters")
optimizer1 = torch.optim.Adam(model1.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=3e-4)
scheduler = LRScheduler(LRConfig())

# actual_batch = 524288 # 2^19
actual_batch = 1024
B, T = 4, 32
accum_iters = actual_batch // (B * T)
data_loader = LangDataLoader(batch_size=B, block_size=T)
data_loader2 = LangDataLoader(batch_size=B, block_size=T)
train_loader, train_loader2 = data_loader.get_train_loader(), data_loader2.get_train_loader()
test_loader = data_loader.get_test_loader()

# ----------------- Training Loop ----------------- #

for curr_iter, ((x, y), (x2, y2)) in enumerate(zip(train_loader, train_loader2)):
    t0 = time.time()
    optimizer1.zero_grad()
    optimizer2.zero_grad()

    loss_accum1 = 0
    loss_accum2 = 0

    # Gradient Accumulation
    for _ in range(accum_iters):
        x, x2 = x.to(device), x2.to(device)
        y, y2 = y.to(device), y2.to(device)

        logits, loss = model1(x, y)
        logits2, loss2 = model2(x2, y2)

        loss /= accum_iters
        loss2 /= accum_iters

        loss_accum1 += loss.detach()
        loss_accum2 += loss2.detach()

        loss.backward()
        loss2.backward()
    
    norm = torch.nn.utils.clip_grad_norm_(model1.parameters(), 1.0) # Prevent shocking the optimization
    norm2 = torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)

    lr = scheduler.get_lr()
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr
    
    optimizer1.step()
    optimizer2.step()
    if device == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()

    print(f"Step {curr_iter:5} | Loss: {loss_accum1:10.6f} | norm: {norm:7.4f} | Time: {t1-t0:6.2f} secs")
    print(f"Step {curr_iter:5} | Loss: {loss_accum2:10.6f} | norm: {norm2:7.4f} | Time: {t1-t0:6.2f} secs")

    











