import torch
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
import argparse
from torchvision import models

# Parse arguments
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR Evaluation')
parser.add_argument('-traind', '--train-dataset', default='cifar10', choices=['stl10', 'cifar10'],
                    help='dataset name')
parser.add_argument('-testd', '--test-dataset', default='cifar100', choices=['stl10', 'cifar10', 'cifar100'],
                    help='dataset name')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')

def get_data_loaders(dataset, download, shuffle=False, batch_size=256):
    data_transform = transforms.ToTensor()

    training_dataset = datasets.CIFAR100('../data', train=True, download=download, transform=data_transform)
    testing_dataset = datasets.CIFAR100('../data', train=False, download=download, transform=data_transform)

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, num_workers=os.cpu_count(),
                              pin_memory=True, shuffle=shuffle, drop_last=False)
    testing_dataloader = DataLoader(testing_dataset, batch_size=2*batch_size, num_workers=os.cpu_count(),
                             pin_memory=True, shuffle=shuffle, drop_last=False)
    return training_dataloader, testing_dataloader

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parser.parse_args()
    
    work_dir = f"/root/Test/SimCLR/{args.arch}_{args.training_dataset}/"
    writer = SummaryWriter(f"/root/Test/SimCLR/simclr/{args.arch}_{args.training_dataset}_eval-on_{args.testing_dataset}/")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_cls = getattr(models, args.arch)
    num_classes = 100
    model = model_cls(pretrained=False, num_classes=num_classes).to(device)
    
    ckpt = torch.load('/root/Test/SimCLR/runs/Jun27_00-08-09_d51a10c7f9a1/checkpoint_0200.pth.tar', map_location=device)
    state_dict = ckpt['state_dict']
    
    for k in list(state_dict.keys()):
        if k.startswith('backbone.') and not k.startswith('backbone.fc'):
            state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]
    
    log = model.load_state_dict(state_dict, strict=False)
    
    training_dataloader, testing_dataloader = get_data_loaders(args.testing_dataset, download=True)
    
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=0.0008)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    epochs = 1000
    n_iter = 0
    for epoch in range(epochs):
        model.train()
        top1_train_acc = 0

        for counter, (x_batch, y_batch) in enumerate(training_dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            
            top1_res = accuracy(logits, y_batch, topk=(1,))
            top1_train_acc += top1_res[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if n_iter % 100 == 0:
                writer.add_scalar('loss', loss.item(), global_step=n_iter)
            n_iter += 1

        top1_train_acc /= (counter + 1)

        model.eval()
        top1_acc = 0
        top5_acc = 0

        with torch.no_grad():
            for counter, (x_batch, y_batch) in enumerate(testing_dataloader):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                logits = model(x_batch)

                top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
                top1_acc += top1[0]
                top5_acc += top5[0]

        top1_acc /= (counter + 1)
        top5_acc /= (counter + 1)

        print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_acc.item()}\tTop1 Test accuracy: {top1_acc.item()}\tTop5 Test accuracy: {top5_acc.item()}")
        writer.add_scalar('acc/train_top1', top1_train_acc.item(), epoch)
        writer.add_scalar('acc/test_top1', top1_acc.item(), epoch)
        writer.add_scalar('acc/test_top5', top5_acc.item(), epoch)