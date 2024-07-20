# This file contains all the experiment details

from dataclasses import dataclass, field
from typing import List
from models import GPT, Lenet, Resnet

@dataclass
class SchedulerConfig:
    gamma: float = 0.1
    milestones: List[int] = field(default_factory=lambda: [32e3, 48e3])

@dataclass
class LeNetConfig:
    # dataset = MNIST.MNISTDataLoader()
    batch_size = 60
    model = Lenet.MNIST_Lenet()
    max_iter = 24e3
    optimizer_name = 'adam'
    lr = 12e-4
    scheduler = None
    dataset = 'mnist'

@dataclass
class ResNet20Config:
    # dataset = CIFAR10.CIFAR10DataLoader
    batch_size = 128
    model = Resnet.Resnet()
    max_iter = 63e3
    optimizer_name = 'sgd'
    momentum = 0.9
    dataset = 'cifar10'
    scheduler = SchedulerConfig()

@dataclass
class ResNet20ConfigStandard(ResNet20Config):
    lr = 0.1
    warmup = 0

@dataclass
class ResNet20ConfigWarmup(ResNet20Config):
    lr = 0.03
    warmup = 30e3

@dataclass
class ResNet20ConfigLow(ResNet20Config):
    lr = 0.01
    warmup = 0

