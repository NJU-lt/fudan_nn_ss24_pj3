import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

def get_dataloader(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    trainset = datasets.CIFAR100(root='./data_new', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root='./data_new', train=False, download=True,transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def initialize_vit(num_classes=100):
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
    

def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)

    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    target_a = target
    target_b = shuffled_target
    return data, target_a, target_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train(model, device, train_loader, writer, optimizer, criterion, epoch, cutmix_prob, alpha=1.0):
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        r = np.random.rand(44)
        # if r < cutmix_prob:
        #     inputs, target_a, target_b, lam = cutmix(inputs, targets, alpha)
        #     output = model(inputs)
        #     loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        # else:
        output = model(inputs)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    writer.add_scalar('Training Loss', running_loss / len(train_loader), epoch)
    print(f'Epoch: {epoch} Training Loss: {running_loss / len(train_loader)}')

def validate(model, writer, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).sum().item()
    
    writer.add_scalar('Validation Accuracy', 100. * correct / total, epoch)
    writer.add_scalar('Validation Loss', test_loss / len(test_loader), epoch)
    print(f'Validation Loss: {test_loss / len(test_loader)} Accuracy: {100. * correct / total}%')
    return test_loss / len(test_loader)

def main(b,lr,e,be,log_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(log_dir=log_dir)

    batch_size = b
    epochs = e
    learning_rate = lr
    
    best_val_loss = float('inf')  # 初始化最佳验证损失
    counter = 0  # 计数器
    cutmix_prob_list = [0,0.1,0.2,0.3,0.4,0.5]
    cutmix_i = 0

    train_loader, test_loader = get_dataloader(batch_size)
    vit_model = initialize_vit().to(device)
    for name, param in vit_model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    for name, param in vit_model.named_parameters():
        if param.requires_grad :
            print(name)

    optimizer = optim.Adam(vit_model.parameters(), lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        cutmix_prob = cutmix_prob_list[cutmix_i]
        
        train(vit_model, device, train_loader, writer, optimizer, criterion, epoch,cutmix_prob=cutmix_prob,alpha=be)
        val_loss = validate(vit_model, writer, device, test_loader, criterion, epoch)
        
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     counter = 0  # 重置计数器
        # elif val_loss >= best_val_loss and cutmix_i < 5:
        #     counter += 1  # 增加计数器
        #     if counter >= 15:
        #         cutmix_i = cutmix_i + 1
        #         print("Cutmix change to " + str(cutmix_prob_list[cutmix_i]))
        #         counter = 0
        # else:
        #     counter += 1  # 增加计数器
        #     if counter >= 15:
        #         print("训练停止" + str(cutmix_prob_list[cutmix_i]))
        #         return 
    
        scheduler.step()
    writer.close()

if __name__ == "__main__":
    batch_size_list = [32]
    lr_list = [3e-5,3e-3]
    epoch_list = [200]
    beta_list = [1]
    for b in batch_size_list:
        for lr in lr_list:
            for e in epoch_list:
                for be in beta_list:
                    log_dir = './runs_resnet18_from_imagenet/cifar100'+ '_' + str(b) + '_' + str(lr) + '_' + str(e)+ '_' + str(be)
                    with torch.cuda.device(2):
                        main(b,lr,e,be,log_dir)