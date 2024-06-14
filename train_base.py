import torch
import copy
from utils import get_hyperparams

experiment = "CIFAR_Resnet20"

# ----------------- Hyperparameters ----------------- #

config = get_hyperparams(experiment)
max_iter = config['iterations']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = config['batch_size']
eval_iter = 50

# -----------------  Necessary Components: Models, Loaders, Optimizers ----------------- #

model, optimizer = config['model'], config['optimizer']
data_loader = config['data_loader']
train_loader, test_loader = data_loader.get_train_loader(), data_loader.get_test_loader()

config = get_hyperparams(experiment)
data_loader2 = config['data_loader']
train_loader2 = data_loader.get_train_loader()

# ----------------- Training Loop ----------------- #

model1 = model
model1 = model1.to(device)
state_dict = copy.deepcopy(model1.state_dict())

model2 = model.to(device)
model2.load_state_dict(state_dict)


optimizer1 = optimizer
optimizer2 = copy.deepcopy(optimizer1)

curr_iter = 0

torch.autograd.set_detect_anomaly(True)
while curr_iter < max_iter:
    for (img, target), (img2, target2) in zip(train_loader, train_loader2):
        img = img.to(device)
        img2 = img2.to(device)

        # Forward Pass
        logits, loss = model1(img, target=target)
        optimizer.zero_grad(set_to_none=True)
        
        # logits2, loss2 = model2(img2, target=target2)
        # optimizer2.zero_grad(set_to_none=True)

        # Backward Pass
        loss.backward()
        optimizer.step()

        # loss2.backward()
        # optimizer2.step()

        # Display Error
        if curr_iter % 1000 == 0:
            print(f"Iteration: {curr_iter}, Loss: {loss.item()}")
        
        curr_iter += 1







