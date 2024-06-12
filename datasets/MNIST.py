from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .base import BaseDataLoader

class MNISTDataLoader(BaseDataLoader):
    def set_transform(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def load_data(self):
        self.set_transform()
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

