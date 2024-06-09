import torch

import torch.nn as nn
from models.Lenet import MNIST_Lenet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import interpolate_weights, save_checkpoint, load_checkpoint
import copy

experiment = "MNIST_Lenet"
# ----------------- Hyperparameters ----------------- #
lr = 12e-4
max_iter = 24000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 60
eval_iter = 50



# ----------------- Training Loop ----------------- #
def mnist_data_loader(batch_size):
    # Transform to convert images to PyTorch tensors and normalize them
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Downloading and loading MNIST test dataset
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

@torch.no_grad()
def estimate_loss(model, mode='eval', copy=False, metric='cross_entropy'):
    dataloader = None
    if mode == 'eval':
        dataloader = test_loader 
    else:
        dataloader = train_loader if not copy else train_loader2
    model.eval()
    losses = torch.zeros(eval_iter)
    for i, (img, target) in enumerate(dataloader):
        if i >= eval_iter:
            break
        img = img.to(device)
        logits, loss = model(img, target)
        if metric == 'cross_entropy':
            losses[i] = loss.item()
        elif metric == 'accuracy':
            pred = logits.argmax(dim=1)
            acc = (pred == target).float().mean()
            losses[i] = acc.item()
        
    model.train()
    return losses.mean()


model1 = MNIST_Lenet()
model1 = model1.to(device)
state_dict1 = copy.deepcopy(model1.state_dict())

model2 = MNIST_Lenet().to(device)
model2.load_state_dict(state_dict1)

train_loader, test_loader = mnist_data_loader(batch_size)
train_loader2, _ = mnist_data_loader(batch_size)

# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model1.parameters(), lr=lr)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr)

# print(len(train_loader))

torch.manual_seed(1337)
curr_iter = 0
while curr_iter < max_iter:
    for (img, target), (img2, target2) in zip(train_loader, train_loader2):
        img = img.to(device)
        img2 = img2.to(device)

        # Forward Pass
        logits, loss = model1(img, target=target)
        optimizer.zero_grad(set_to_none=True)
        
        logits2, loss2 = model2(img2, target=target2)
        optimizer2.zero_grad(set_to_none=True)

        # Backward Pass
        loss.backward()
        optimizer.step()

        loss2.backward()
        optimizer2.step()

        # Display Error
        if curr_iter % 1000 == 0:
            print(f"Iter: {curr_iter} (Model 1), TrainLoss: {estimate_loss(model1, 'train')}, EvalLoss: {estimate_loss(model1, 'eval')}")
            print(f"Iter: {curr_iter} (Model 2), TrainLoss: {estimate_loss(model2, 'train', True)}, EvalLoss: {estimate_loss(model2, 'eval')}")
        
        curr_iter += 1


print(f"Model 1 Accuracy: {estimate_loss(model1, 'eval', metric='accuracy')}")
print(f"Model 2 Accuracy: {estimate_loss(model2, 'eval', copy=True, metric='accuracy')}")

save_checkpoint(model1, curr_iter, 'MNIST_Lenet/model1')
save_checkpoint(model2, curr_iter, 'MNIST_Lenet/model2')


interpolated_model = interpolate_weights(model1, model2, MNIST_Lenet(), 0.5, device=device)
print(f"Interpolated Model Accuracy: {estimate_loss(interpolated_model, 'eval', metric='accuracy')}")


# Interpolation
res = 30

alphas = torch.linspace(0, 1, res)
error_rates = torch.zeros((2, res))

for i, alpha in enumerate(alphas):
    interpolated_model = interpolate_weights(model1, model2, MNIST_Lenet(), alpha, device=device)
    acc = estimate_loss(interpolated_model, 'eval', metric='accuracy')
    error_rates[0, i] = 1 - acc
    acc = estimate_loss(interpolated_model, 'train', metric='accuracy')
    error_rates[1, i] = 1 - acc

error_rates *= 100

import matplotlib.pyplot as plt
plt.plot(alphas, error_rates[0, :], 'r') # Eval
plt.plot(alphas, error_rates[1, :], 'b') # Train
plt.xlabel('Interpolation')
plt.ylabel('Error (%)')
plt.ylim(0, 100)
plt.title(experiment)

plt.grid(True)  # Enable both major and minor grid lines
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

plt.show()
plt.savefig('MNIST_Lenet/interpolation.png')


