import torch
import time
import copy
import matplotlib.pyplot as plt
from training_script.utils import get_hyperparams, estimate_loss, interpolate_weights, save_checkpoint, load_checkpoint
import os

experiment = "MinGPT_Shakespeare"

# ----------------- Hyperparameters ----------------- #

config = get_hyperparams(experiment)
max_iter = config['iterations']
batch_size = config['batch_size']
eval_iter = 50

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'

print(f"Device: {device}")
# -----------------  Necessary Components: Models, Loaders, Optimizers ----------------- #

model1, optimizer1 = config['model'], config['optimizer']
model1 = model1.to(device)
data_loader = config['data_loader']
train_loader, test_loader = data_loader.get_train_loader(), data_loader.get_test_loader()
baseline = copy.deepcopy(model1)
scheduler1 = config['scheduler']

print(f"Baseline is {type(baseline)}, with {sum(p.numel() for p in baseline.parameters() if p.requires_grad)} parameters.")

config = get_hyperparams(experiment)
data_loader2 = config['data_loader']
train_loader2 = data_loader2.get_train_loader()
model2, optimizer2 = config['model'], config['optimizer']
model2.load_state_dict(copy.deepcopy(model1.state_dict()))
model2 = model2.to(device)
scheduler2 = config['scheduler']


# ----------------- Training Loop ----------------- #

def train():
    curr_iter = 0
    torch.autograd.set_detect_anomaly(True)
    while curr_iter < max_iter:
        for (img, target), (img2, target2) in zip(train_loader, train_loader2):
            # print(curr_iter)
            img = img.to(device)
            img2 = img2.to(device)
            target = target.to(device)
            target2 = target2.to(device)
            t0 = time.time()
            # Forward Pass
            logits, loss = model1(img, target=target)
            optimizer1.zero_grad(set_to_none=True)
            
            logits2, loss2 = model2(img2, target=target2)
            optimizer2.zero_grad(set_to_none=True)

            # Backward Pass
            loss.backward()
            optimizer1.step()

            loss2.backward()
            optimizer2.step()
            t1 = time.time()

            # Display Error
            print(f"Iter: {curr_iter} (Model 1), TrainLoss: {estimate_loss(model1, train_loader, eval_iter, device)}, EvalLoss: {estimate_loss(model1, test_loader, eval_iter, device)}")
            print(f"Iter: {curr_iter} (Model 2), TrainLoss: {estimate_loss(model2, train_loader2, eval_iter, device)}, EvalLoss: {estimate_loss(model2, test_loader, eval_iter, device)}")
            print(f"Time taken: {t1 - t0}\n")
            curr_iter += 1
            if curr_iter >= max_iter:
                break
        scheduler1.step()
        scheduler2.step()

# ----------------- Checkpointing ----------------- #
user_input = input("Do you want to load checkpoints? (y/n): ")
if not os.path.isdir(experiment) or not os.listdir(experiment) or user_input == 'n':
    train()
    save_checkpoint(model1, max_iter, experiment + '/model1')
    save_checkpoint(model2, max_iter, experiment + '/model2')
else:
    print("Loading Checkpoints")
    model1, optimizer1 = load_checkpoint(model1, experiment + '/model1')
    model2, optimizer2 = load_checkpoint(model2, experiment + '/model2')

# ----------------- Save Model ----------------- #


# ----------------- Linear Interpolation ----------------- #

res = 30 # Resolution of the interpolation

alphas = torch.linspace(0, 1, res)
error_rates = torch.zeros((2, res))

for i, alpha in enumerate(alphas):
    interpolated_model = interpolate_weights(model1, model2, baseline, alpha, device=device)
    acc = estimate_loss(interpolated_model, test_loader, eval_iter, device, metric='accuracy')
    error_rates[0, i] = 1 - acc
    acc = estimate_loss(interpolated_model, train_loader, eval_iter, device, metric='accuracy')
    error_rates[1, i] = 1 - acc

error_rates *= 100

plt.plot(alphas, error_rates[0, :], 'r') # Eval
plt.plot(alphas, error_rates[1, :], 'b') # Train
plt.legend(['Eval', 'Train'])
plt.xlabel('Interpolation')
plt.ylabel('Error (%)')
plt.ylim(0, 100)
plt.title(experiment)

plt.grid(True)  # Enable both major and minor grid lines
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
plt.savefig(f"{experiment}_interpolation.png")
plt.show()

