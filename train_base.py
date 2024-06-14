import torch
import copy
import matplotlib.pyplot as plt
from utils import get_hyperparams, estimate_loss, interpolate_weights

experiment = "MNIST_Lenet"

# ----------------- Hyperparameters ----------------- #

config = get_hyperparams(experiment)
max_iter = config['iterations']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = config['batch_size']
eval_iter = 50

# -----------------  Necessary Components: Models, Loaders, Optimizers ----------------- #

model1, optimizer1 = config['model'], config['optimizer']
data_loader = config['data_loader']
train_loader, test_loader = data_loader.get_train_loader(), data_loader.get_test_loader()
baseline = copy.deepcopy(model1)

config = get_hyperparams(experiment)
data_loader2 = config['data_loader']
train_loader2 = data_loader2.get_train_loader()
model2, optimizer2 = config['model'], config['optimizer']

# ----------------- Training Loop ----------------- #

curr_iter = 0

torch.autograd.set_detect_anomaly(True)
while curr_iter < max_iter:
    for (img, target), (img2, target2) in zip(train_loader, train_loader2):
        img = img.to(device)
        img2 = img2.to(device)

        # Forward Pass
        logits1, loss1 = model1(img, target=target)
        optimizer1.zero_grad(set_to_none=True)
        
        logits2, loss2 = model2(img2, target=target2)
        optimizer2.zero_grad(set_to_none=True)

        # Backward Pass
        loss1.backward()
        optimizer1.step()

        loss2.backward()
        optimizer2.step()

        # Display Error
        if curr_iter % 1000 == 0:
            print("Iteration: ", curr_iter)
            print("Estimate Loss Model 1: ", estimate_loss(model1, test_loader, eval_iter, device, metric='cross_entropy'))
            print("Estimate Loss Model 2: ", estimate_loss(model2, test_loader, eval_iter, device, metric='cross_entropy'))        
        curr_iter += 1

# ----------------- Linear Interpolation ----------------- #

res = 30 # Resolution of the interpolation

alphas = torch.linspace(0, 1, res)
error_rates = torch.zeros((2, res))


for i, alpha in enumerate(alphas):
    interpolated_model = interpolate_weights(model1, model2, baseline, alpha, device=device)
    acc = estimate_loss(interpolated_model, 'eval', metric='accuracy')
    error_rates[0, i] = 1 - acc
    acc = estimate_loss(interpolated_model, 'train', metric='accuracy')
    error_rates[1, i] = 1 - acc

error_rates *= 100

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

