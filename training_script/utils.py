import torch
import os
import json
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR

import datasets.MNIST
import datasets.CIFAR10
import datasets.Langdata
import models.GPT
import models.Lenet
import models.Resnet

class NoOpScheduler:
    def step(self):
        pass
def get_hyperparams(experiment_name, config_file = "config.json"):
    with open(config_file, 'r') as f:
        configs = json.load(f)
    for config in configs:
        if config["name"] == experiment_name:
            vocab_size = None
            GPT_Params, n, input_file, block_size = None, None, None, None
            if "GPT_params" in config:
                GPT_Params = config["GPT_params"]
                block_size = GPT_Params["block_size"]
                n = config["n"]
                input_file = config["input_file"]
                vocab_size = get_vocab_size(input_file)

                        
            config["model"] = get_model(config["model"], vocab_size, **GPT_Params)
            config["optimizer"] = get_optimizer(config["optimizer"]['name'], config["model"], config["optimizer"]["lr"])
            config["data_loader"] = get_loader(config["data_loader"], config["batch_size"], block_size=block_size, n=n, input_file=input_file)
            if config["scheduler"] is not None:
                if config["variant"] == "standard":
                    config["scheduler"] = MultiStepLR(config["optimizer"], milestones=config["scheduler"]["milestones"], gamma=config["scheduler"]["gamma"])
                elif config["variant"] == "low":
                    config["scheduler"] = NoOpScheduler()
                elif config["variant"] == "warmup":
                    config["scheduler"] = get_warmup_scheduler(config["optimizer"], num_warmup_steps=config["scheduler"]["warmup_steps"])
            else:
                config["scheduler"] = NoOpScheduler()
            break

    return config
def get_warmup_scheduler(optimizer, num_warmup_steps=30000):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler
def get_model(model_name, vocab_size=None, **kwargs):
    if model_name == 'GPT':
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        if vocab_size is None:
            raise ValueError("Vocab size required for GPT model")
        return models.GPT.BigramLanguageModel(vocab_size, device=device, **kwargs)
    elif model_name == "Lenet":
        return models.Lenet.MNIST_Lenet()
    elif model_name == "Resnet20":
        return models.Resnet.Resnet20()
    
    print(f"Model not found: {model_name}")
    raise ValueError("Model not found")

def get_optimizer(optimizer_name, model, lr):
    if optimizer_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr)
    
    print(f"Optimizer not found: {optimizer_name}")
    raise ValueError("Optimizer not found")

def get_loader(data_loader_name, batch_size, **kwargs):
    if data_loader_name == 'MNIST':
        return datasets.MNIST.MNISTDataLoader(batch_size)
    elif data_loader_name== 'CIFAR10':
        return datasets.CIFAR10.CIFAR10DataLoader(batch_size)
    elif data_loader_name == 'Lang':
        return datasets.Langdata.LangDataLoader(batch_size, kwargs["block_size"], kwargs["n"], kwargs["input_file"])
    
    print(f"Data Loader not found: {data_loader_name}")
    raise ValueError("Data Loader not found")

@torch.no_grad()
def estimate_loss(model, dataloader, eval_iter, device, metric='cross_entropy'):
    model.eval()
    losses = torch.zeros(eval_iter, device=device)
    for i, (img, target) in enumerate(dataloader):
        if i >= eval_iter:
            break
        img = img.to(device)
        target = target.to(device)
        logits, loss = model(img, target)
        if metric == 'cross_entropy':
            losses[i] = loss.item()
        elif metric == 'accuracy':
            pred = logits.argmax(dim=-1)
            acc = (pred == target).float().mean()
            losses[i] = acc.item()
    model.train()
    return losses.mean().item()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_vocab_size(filename='input.txt'):
    with open(filename, 'r') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    return vocab_size

def interpolate_weights(model1, model2, baseline, alpha, device='cpu'):
    interpolated_model = baseline.to(device)
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    interpolated_state_dict = {}

    for key in state_dict1.keys():
        interpolated_state_dict[key] = alpha * state_dict1[key] + (1 - alpha) * state_dict2[key]

    interpolated_model.load_state_dict(interpolated_state_dict)
    return interpolated_model

def save_checkpoint(model, epoch, checkpoint_dir, optimizer=None):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint = {
        'model': model.state_dict(),
        'epoch': epoch,
    }
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    filename = os.path.join(checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(epoch))
    torch.save(checkpoint, filename)

def load_checkpoint(model, checkpoint_dir, optimizer=None):
    checkpoint_files = os.listdir(checkpoint_dir)
    if len(checkpoint_files) == 0:
        return model, optimizer

    checkpoint_files = sorted(checkpoint_files)
    last_checkpoint = checkpoint_files[-1]
    checkpoint = torch.load(os.path.join(checkpoint_dir, last_checkpoint))
    model.load_state_dict(checkpoint['model'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return model, optimizer