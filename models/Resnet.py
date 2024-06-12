import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) # With batch norm, no need for bias
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # As mentioned in He et al. 2015, we need shortcuts to be of the same dimension as the output
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class Resnet(nn.Module):
    def __init__(self, initial_features, num_blocks, num_classes=10) -> None:
        super().__init__()
        self.in_features = initial_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=initial_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(initial_features),
        )

        self.layer1 = self._make_layer(num_blocks[0], initial_features, 1)
        self.layer2 = self._make_layer(num_blocks[1], initial_features*2, 2)
        self.layer3 = self._make_layer(num_blocks[2], initial_features*4, 2)
        self.fc = nn.Linear(initial_features*4, num_classes) # CIFAR-10

    
    def _make_layer(self, num_blocks, out_channels, stride):
        layers = [Block(self.in_features, out_channels, stride)]
        self.in_features = out_channels
        for _ in range(num_blocks - 1):
            layers.append(Block(self.in_features, out_channels, 1))
            self.in_features = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x


def Resnet20():
    return Resnet(16, [3, 3, 3])


if __name__ == '__main__':
    model = Resnet20()
    print(f"Resnet20 has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters.")

