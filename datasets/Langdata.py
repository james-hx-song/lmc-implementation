import torch
from .base import BaseDataLoader
from torch.utils.data import Dataset, DataLoader

class LangDataset(Dataset):
    def __init__(self, block_size, train, n=0.9, input_file="input.txt"):
        with open(input_file, 'r') as f:
            self.text = f.read()
        self.train = train
        chars = sorted(list(set(self.text)))
        self.block_size = block_size
        self.itos = {i:ch for i, ch in enumerate(chars)}
        self.stoi = {ch:i for i, ch in enumerate(chars)}

        self.data = torch.tensor([self.stoi[c] for c in self.text], dtype=torch.long)
        self.train_data = self.data[:int(n*len(self.data))]
        self.val_data = self.data[int(n*len(self.data)):]

    def __len__(self):
        if self.train:
            return len(self.train_data) - self.block_size
        return len(self.val_data) - self.block_size
    
    def __getitem__(self, idx):
        data = self.train_data if self.train else self.val_data
        
        return data[idx:idx+self.block_size], data[idx+1:idx+self.block_size+1]

class LangDataLoader(BaseDataLoader):

    def __init__(self, batch_size, block_size, n=0.9, input_file="input.txt"):
        super().__init__(batch_size)
        self.block_size = block_size
        self.input_file = input_file
        self.n = n
        self.input_file = input_file
    def set_transform(self):
        pass

    def load_data(self):
        train_dataset = LangDataset(self.block_size, train=True, n=self.n, input_file=self.input_file)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        test_dataset = LangDataset(self.block_size, train=False, n=self.n, input_file=self.input_file)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

