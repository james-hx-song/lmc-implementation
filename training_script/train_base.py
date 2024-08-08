import torch
import time
import copy
from training_script.utils import get_hyperparams, estimate_loss, interpolate_weights, save_checkpoint, load_checkpoint, config_dict, visualize_interpolation
import os

experiment = "cifar_resnet"

device = 'cpu'
device_type = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    device_type = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
# device = 'cpu'
print(f"Device: {device}")
# -----------------  Necessary Components: Models, Loaders, Optimizers ----------------- #
max_iter = config_dict[experiment].max_iter
model1, optimizer1, data_loader, scheduler = get_hyperparams(experiment)
model2, optimizer2, data_loader2, _ = get_hyperparams(experiment)
model2.load_state_dict(model1.state_dict())
train_loader, train_loader2  = data_loader.get_train_loader(), data_loader2.get_train_loader()
test_loader = data_loader.get_test_loader()

model1 = torch.compile(model1)
model2 = torch.compile(model2)
model1.to(device)
model2.to(device)

eval_iter = 50
baseline = copy.deepcopy(model1)

print(f"Model: {type(model1).__name__}")

# if scheduler is not None:
#     print(f"Config LR: {scheduler.lr}")
# ----------------- Training Loop ----------------- #
def train(curr_iter=0):
    prev_lr = -1
    torch.autograd.set_detect_anomaly(True)
    # model1 = torch.compile(model1)
    # model2 = torch.compile(model2)
    while curr_iter < max_iter:
        for (img, target), (img2, target2) in zip(train_loader, train_loader2):
            # print(curr_iter)
            img, img2 = img.to(device), img2.to(device)
            target, target2 = target.to(device), target2.to(device)
            t0 = time.time()

            # Set to zero grad
            optimizer1.zero_grad(set_to_none=True)
            optimizer2.zero_grad(set_to_none=True)

            # Forward Pass
            # with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model1(img, target=target)
            logits2, loss2 = model2(img2, target=target2)

            # Backward Pass
            loss.backward()
            loss2.backward()

            # Update LR
            if scheduler is not None:
                lr = scheduler.get_lr(curr_iter)
                if prev_lr != lr: # Save time
                    # print(f"{lr:.10f}")
                    for param_group in optimizer1.param_groups:
                        param_group['lr'] = lr
                    for param_group in optimizer2.param_groups:
                        param_group['lr'] = lr
                prev_lr = lr

            optimizer1.step()
            optimizer2.step()
            t1 = time.time()

            # Display Error
            print(f"Iter: {curr_iter} (Model 1), TrainLoss: {loss.item()}")
            print(f"Iter: {curr_iter} (Model 2), TrainLoss: {loss2.item()}")
            if curr_iter % 1000 == 0:
                # Every 1000 Iteration, we run an iterated evaluation on the loss to get a better estimate
                print(f"\nIter: {curr_iter} (Model 1), TrainLoss: {estimate_loss(model1, train_loader, eval_iter, device)}, EvalLoss: {estimate_loss(model1, test_loader, eval_iter, device)}")
                print(f"Iter: {curr_iter} (Model 2), TrainLoss: {estimate_loss(model2, train_loader2, eval_iter, device)}, EvalLoss: {estimate_loss(model2, test_loader, eval_iter, device)}")
            if curr_iter % 2000 == 0:
                save_checkpoint(model1, curr_iter, experiment + '/model1', optimizer1)
                save_checkpoint(model2, curr_iter, experiment + '/model2', optimizer2)
            print(f"Time taken: {t1 - t0}\n")
            curr_iter += 1
            if curr_iter >= max_iter:
                break

# ----------------- Checkpointing ----------------- #
if os.path.isdir(experiment) and os.listdir(experiment):
    print("Checkpoints Found")
    user_input = input("Do you want to load checkpoints? (y/n): ")
    if user_input == 'y':
        print("Loading Checkpoints")
        iter = int(input("Enter Iteration to Load: "))
        model1, optimizer1, curr_iter = load_checkpoint(model1, experiment + '/model1', iterations=iter, device=device)
        model2, optimizer2, _ = load_checkpoint(model2, experiment + '/model2', iterations=iter, device=device)

        user_input = input("Do you want to continue training? (y/n): ")
        if user_input == 'y':
            print(f"Continuing Training from Iteration {curr_iter}")
            train(curr_iter)
            save_checkpoint(model1, max_iter, experiment + '/model1', optimizer1)
            save_checkpoint(model2, max_iter, experiment + '/model2', optimizer2)
    else:
        train()
        save_checkpoint(model1, max_iter, experiment + '/model1')
        save_checkpoint(model2, max_iter, experiment + '/model2')
else:
    train()
    save_checkpoint(model1, max_iter, experiment + '/model1')
    save_checkpoint(model2, max_iter, experiment + '/model2')


# ----------------- Linear Interpolation ----------------- #

res = 30 # Resolution of the interpolation
alphas = torch.linspace(0, 1, res)
error_rates = torch.zeros((2, res))
eval_iter = 'all'
for i, alpha in enumerate(alphas):
    interpolated_model = interpolate_weights(model1, model2, baseline, alpha, device=device)
    acc = estimate_loss(interpolated_model, test_loader, eval_iter, device, metric='accuracy')
    error_rates[0, i] = 1 - acc
    acc = estimate_loss(interpolated_model, train_loader, eval_iter, device, metric='accuracy')
    error_rates[1, i] = 1 - acc

visualize_interpolation(alphas, error_rates, experiment)
