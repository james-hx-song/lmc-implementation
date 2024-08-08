import torch
import os
import json
import matplotlib.pyplot as plt
import random
from typing import Union

from datasets import CIFAR10, MNIST, Langdata
import config.Exp_Config as config
from scheduler.scheduler import LRScheduler

config_dict = dict(
    mnist_lenet=config.LeNetConfig(),
    cifar_resnet=config.ResNet20ConfigStandard(),
    cifar_resnet_warmup=config.ResNet20ConfigWarmup(),
    cifar_resnet_low=config.ResNet20ConfigLow(),
    cifar_vgg=config.VGG16ConfigStandard(),
    cifar_vgg_warmup=config.VGG16ConfigWarmup(),
    cifar_vgg_low=config.VGG16ConfigLow(),
)

datasets = dict(
    mnist=MNIST.MNISTDataLoader,
    cifar10=CIFAR10.CIFAR10DataLoader,
    langdata=Langdata.LangDataLoader
)

def get_hyperparams(experiment):
    if experiment not in config_dict:
        raise ValueError(f"Experiment {experiment} not found in config.")
    config = config_dict[experiment]
    model = config.model
    if config.optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)

    scheduler = None if config.scheduler is None else LRScheduler(config)

    data_loader = datasets[config.dataset](batch_size=config.batch_size)

    return model, optimizer, data_loader, scheduler

@torch.no_grad()
def estimate_loss(model, dataloader, eval_iter: Union[str, int], device, metric='cross_entropy'):
    model.eval()
    losses = torch.zeros(len(dataloader), device=device)
    if eval_iter == 'all':
        for i, (img, target) in enumerate(dataloader):
            img = img.to(device)
            target = target.to(device)
            print("Forward Pass")
            logits, loss = model(img, target)
            print("Forward Pass Done")
            if metric == 'cross_entropy':
                losses[i] = loss.item()
            elif metric == 'accuracy':
                pred = logits.argmax(dim=-1)
                acc = (pred == target).float().mean()
                losses[i] = acc.item()
    elif isinstance(eval_iter, int):
        dataloaderlist = list(dataloader)
        for i in range(eval_iter):
            img, target = random.choice(dataloaderlist)
            img = img.to(device)
            target = target.to(device)
            print("Forward Pass")
            logits, loss = model(img, target)
            print("Forward Pass Done")
            if metric == 'cross_entropy':
                losses[i] = loss.item()
            elif metric == 'accuracy':
                pred = logits.argmax(dim=-1)
                acc = (pred == target).float().mean()
                losses[i] = acc.item()
    else:
        raise ValueError("Invalid eval_iter value. Must be 'all' or an integer.")
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

def visualize_interpolation(alphas, error_rates, experiment):
    error_rates *= 100

    if not os.path.exists("process_imgs"):
        os.makedirs("process_imgs")
    plt.plot(alphas, error_rates[0, :], 'r') # Eval
    plt.plot(alphas, error_rates[1, :], 'b') # Train
    plt.legend(['Eval', 'Train'])
    plt.xlabel('Interpolation')
    plt.ylabel('Error (%)')
    plt.ylim(0, 100)
    plt.title(experiment)

    plt.grid(True)  # Enable both major and minor grid lines
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    plt.savefig(f"process_imgs/{experiment}_interpolation.png")
    plt.show()

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

def load_checkpoint(model, checkpoint_dir, iterations=None, optimizer=None, device='cpu'):
    checkpoint_files = os.listdir(checkpoint_dir)
    if len(checkpoint_files) == 0:
        return model, optimizer

    checkpoint_files = sorted(checkpoint_files)
    last_checkpoint = checkpoint_files[-1]
    if iterations is None:
        iterations = int(last_checkpoint.split('=')[1].split('.')[0])
    dir = os.path.join(checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(iterations))
    print(dir)
    checkpoint = torch.load(dir, map_location=torch.device(device))
    # for key, value in checkpoint['model'].items():
    #     print(key)
    model.load_state_dict(checkpoint['model'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return model, optimizer, iterations