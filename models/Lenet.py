import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- Model Architecture ----------------- #
class MNIST_Lenet(nn.Module):

  def __init__(self, hidden_sizes=[300, 100], output_size=10) -> None:
    super().__init__()

    # Lenet Architecture

    fc_layers = []
    curr_size = 28*28 # Mnist 
    for size in hidden_sizes:
      fc_layers.append(nn.Linear(curr_size, size))
      fc_layers.append(nn.ReLU())
      curr_size = size
    
    self.fc = nn.Linear(curr_size, output_size)
    self.fc_layers = nn.Sequential(*fc_layers)


  def forward(self, x):
    print(x.shape)
    x = x.view(x.size(0), -1)
    x = self.fc_layers(x)
    return self.fc(x)

if __name__ == '__main__':
  model = MNIST_Lenet()

  print(f"Lenet has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters.")



