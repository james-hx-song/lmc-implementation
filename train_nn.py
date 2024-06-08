import torch

import torch.nn as nn
from models.Lenet import MNIST_Lenet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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
def estimate_loss(mode='eval'):
    dataloader = None
    if mode == 'eval':
        dataloader = test_loader
    else:
        dataloader = train_loader
    model.eval()
    losses = torch.zeros(eval_iter)
    for i, (img, target) in enumerate(dataloader):
        if i >= eval_iter:
            break
        img = img.to(device)
        logits, loss = model(img, target)
        losses[i] = loss.item()
    model.train()
    return losses.mean()


model = MNIST_Lenet()
model = model.to(device)

train_loader, test_loader = mnist_data_loader(batch_size)
train_loader2, _ = mnist_data_loader(batch_size)

# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print(len(train_loader))

torch.manual_seed(1337)
curr_iter = 0
while curr_iter < max_iter:
    for img, target in train_loader:
        img = img.to(device)

        # Forward Pass
        logits, loss = model(img, target=target)
        optimizer.zero_grad(set_to_none=True)

        # Backward Pass
        loss.backward()
        optimizer.step()

        # Display Error
        if curr_iter % 1000 == 0:
            print(f"Iter: {curr_iter}, TrainLoss: {estimate_loss('train')}, EvalLoss: {estimate_loss('eval')}")
        
        curr_iter += 1








