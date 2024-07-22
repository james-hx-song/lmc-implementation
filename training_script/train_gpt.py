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
model1.to(device)

model2 = GPT(GPTConfig())
model2.to(device)

print(f"Model: {type(model1).__name__}")
print(f"with {sum(p.numel() for p in model1.parameters() if p.requires_grad)} parameters")
optimizer1 = torch.optim.Adam(model1.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
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
for curr_iter, (x, y) in enumerate(train_loader):
    t0 = time.time()
    optimizer1.zero_grad()

    loss_accum = 0
    for _ in range(accum_iters):
        x, y = x.to(device), y.to(device)
        # with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model1(x, y)  
        loss /= accum_iters # Gradient Accumulation Correction (1/N)
        loss_accum += loss.detach() # Accumulate loss. Detach the tensor from the graph
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model1.parameters(), 1.0) # Prevent shocking the optimization

    # Update Learning Rate
    lr = scheduler.get_lr(curr_iter)
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr

    optimizer1.step()
    if device == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()

    # with open(debug_file, "w") as f:
    #     f.write(f"Step {i:5} | Loss: {loss_accum:10.6f} | norm: {norm:7.4f} | Time: {t1-t0:6.2f} secs\n")
    print(f"Step {curr_iter:5} | Loss: {loss_accum:10.6f} | norm: {norm:7.4f} | Time: {t1-t0:6.2f} secs")


for curr_iter, (x, y) in enumerate(train_loader2):
    t0 = time.time()
    optimizer2.zero_grad()

    loss_accum = 0
    for _ in range(accum_iters):
        x, y = x.to(device), y.to(device)
        # with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model2(x, y)  
        loss /= accum_iters # Gradient Accumulation Correction (1/N)
        loss_accum += loss.detach() # Accumulate loss. Detach the tensor from the graph
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0) # Prevent shocking the optimization

    # Update Learning Rate
    lr = scheduler.get_lr(curr_iter)
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr

    optimizer2.step()
    if device == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()

    # with open(debug_file, "w") as f:
    #     f.write(f"Step {i:5} | Loss: {loss_accum:10.6f} | norm: {norm:7.4f} | Time: {t1-t0:6.2f} secs\n")
    print(f"Step {curr_iter:5} | Loss: {loss_accum:10.6f} | norm: {norm:7.4f} | Time: {t1-t0:6.2f} secs")

    




