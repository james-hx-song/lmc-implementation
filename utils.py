import torch
import os

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