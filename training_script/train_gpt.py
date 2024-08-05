# GPT is written as a separate file because its training loop and configurations are significantly different from other models, 
# due to gradient accumulation and clipping grad norm
import torch
import time
import copy
import random
import sys

# Setup / Config
device = 'cpu'
device_type = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    device_type = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f"Using {device}")
device = 'cpu'


# -----------------  Necessary Components: Models, Loaders, Optimizers ----------------- #
from models.GPT import GPT, GPT2Config, ToyGPTConfig, MinGPTConfig
from scheduler.GPTScheduler import LRScheduler, LRConfig
from datasets.Langdata import LangDataLoader
from training_script.utils import count_parameters, estimate_loss

model_config = ToyGPTConfig(vocab_size=50304)

model1 = GPT(model_config)
model2 = copy.deepcopy(model1)

model1.to(device)
model2.to(device)

print(f"Model: {type(model1).__name__}, with {count_parameters(model1)} parameters.")
optimizer1 = torch.optim.Adam(model1.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
scheduler = LRScheduler(LRConfig())

actual_batch_size = model_config.batch_size
B, T = 4, model_config.block_size

accum_iters = actual_batch_size // B
data_loader = LangDataLoader(batch_size=B, block_size=T)
data_loader2 = LangDataLoader(batch_size=B, block_size=T)
train_loader, train_loader2 = data_loader.get_train_loader(), data_loader2.get_train_loader()


test_loader = data_loader.get_test_loader()
eval_iters = 50

print(f"Loading with each accumulation batch {B} and sequence length {T}.")
print(f"There are a total of {len(train_loader)} training batches.")

# ----------------- Training Loop ----------------- #

def train(model, optimizer, train_loader, test_loader=None, epochs=5):
    model = torch.compile(model)
    for _ in range(epochs):
        for curr_iter, (x, y) in enumerate(train_loader):
            t0 = time.time()
            optimizer.zero_grad()

            loss_accum = 0
            for _ in range(accum_iters):
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)  
                loss /= accum_iters # Gradient Accumulation Correction (1/N)
                loss_accum += loss.detach() # Accumulate loss. Detach the tensor from the graph
                loss.backward()

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Prevent shocking the optimization

            # Update Learning Rate
            lr = scheduler.get_lr(curr_iter)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize() # wait for the GPU to finish work
            t1 = time.time()

            print(f"Step {curr_iter:5} | Loss: {loss_accum:10.6f} | norm: {norm:7.4f} | Time: {(t1-t0)*1e3:10.2f} ms")

            if curr_iter % 200 == 0 and test_loader is not None:
                train_loss = estimate_loss(model, train_loader, eval_iters, device)
                test_loss = estimate_loss(model, test_loader, eval_iters, device)
                print(f"\nIter: {curr_iter} (Model 1), TrainLoss: {train_loss}, EvalLoss: {test_loss}\n")

torch.set_float32_matmul_precision('high')
train(model1, optimizer1, train_loader, test_loader)



