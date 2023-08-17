import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder
from torchvision import transforms

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import dataset
from model import Model


class Encoder(nn.Module):
    def __init__(self, dataset, pretrained_path=None):
        super(Encoder, self).__init__()

        # encoder
        model = Model(dataset)
        if pretrained_path is not None:
            model.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
            
        self.f = model.f
        self.f.fc = nn.Identity()

    def forward(self, x):
        out = self.f(x)
        return out

class FC(torch.nn.Module):
    def __init__(self, num_class):
        super(FC, self).__init__()
        self.linear = torch.nn.Linear(2048, num_class)
        
    def forward(self, x):
        return self.linear(x)

def get_features_from_encoder(encoder, loader, device):
    
    x_train = []
    y_train = []

    # get the features from the pre-trained model
    for i, (x, y) in tqdm(enumerate(loader), total=len(loader), desc='Saving features'):
        with torch.no_grad():
            x = x.to(device)
            feature_vector = encoder(x)
            x_train.extend(feature_vector.cpu())
            y_train.extend(y.numpy())

            
    x_train = torch.stack(x_train)
    y_train = torch.tensor(y_train)
    return x_train, y_train

def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test):

    train = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

    test = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False)
    return train_loader, test_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labeled_ratio', type=float, default=0.01, help='Ratio of labeled data')
    parser.add_argument('--dataset', default='tinyimagenet', type=str, help='Dataset')
    parser.add_argument('--algo', type=str, default='myol')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--checkpoint', type=int, default=500)
    parser.add_argument('--optim', default='sgd', type=str, help='Optimizer')
    parser.add_argument('--lr', default=0.05, type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.dataset == 'tinyimagenet':
        batch_size = 64
    else:
        batch_size = 32
    args.epochs = 20
        
    model_name = f'{args.algo}_batch{args.batch_size}_{args.optim}{args.lr}_{args.seed}'
    model_path = f'results/pretrain/{args.dataset}/{args.algo}/{model_name}_{args.checkpoint}.pth'
    result_path = f'results/downstream/{args.dataset}/{args.algo}'
    semisup_result_path = f'{result_path}/semisup{args.labeled_ratio}_{model_name}_statistics_{args.checkpoint}.csv'

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    print(model_name, args.checkpoint)
    if os.path.exists(semisup_result_path):
        print('Already done')
        import sys
        sys.exit()
        
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = dataset.CIFAR10.get_transform(train=False)
        train_data = CIFAR10(root='./data', train=True, transform=train_transform, download=True)
        test_data = CIFAR10(root='./data', train=False, transform=test_transform, download=True)
        class_idx = np.array(train_data.targets)
    elif args.dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])
        test_transform = dataset.CIFAR100.get_transform(train=False)
        train_data = CIFAR100(root='./data', train=True, transform=train_transform, download=True)
        test_data = CIFAR100(root='./data', train=False, transform=test_transform, download=True)
        class_idx = np.array(train_data.targets)
    elif args.dataset == 'stl10':
        train_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4467, 0.4398, 0.4066], [0.2603, 0.2566, 0.2713])
        ])
        test_transform = dataset.STL10.get_transform(train=False)
        train_data = STL10(root='./data', split='train', transform=train_transform, download=True)
        test_data = STL10(root='./data', split='test', transform=test_transform, download=True)
        class_idx = np.array(train_data.labels)
    elif args.dataset == 'tinyimagenet':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        ])
        test_transform = dataset.TinyImageNet.get_transform(train=False)
        train_data = ImageFolder(root='./data/tiny-imagenet-200/train', transform=train_transform)
        test_data = ImageFolder(root='./data/tiny-imagenet-200/val', transform=test_transform)
        class_idx = np.array(train_data.targets)

    num_class = len(train_data.classes)
    class_idx = np.array([np.where(class_idx == i)[0] for i in range(num_class)])

    indices = []
    for i in range(num_class):
        idx = class_idx[i]
        np.random.shuffle(idx)
        indices.extend(idx[:int(len(idx) * args.labeled_ratio)])
        
    train_data = torch.utils.data.Subset(train_data, indices)
    train_loader = DataLoader(train_data, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(args.dataset, pretrained_path=model_path).to(device)

    fc = FC(num_class=num_class)
    fc = fc.to(device)
    
    # vicreg
    optimizer = torch.optim.SGD([
        {'params': encoder.parameters(), 'lr': 0.01 if args.labeled_ratio == 0.1 else 0.03},
        {'params': fc.parameters(), 'lr': 0.1 if args.labeled_ratio == 0.1 else 0.08}],
        momentum=0.9,
        weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*len(train_loader), eta_min=0, last_epoch=-1)
    criterion = torch.nn.CrossEntropyLoss()
    
    eval_every_n_epochs = 1
    test_results = {'test_acc@1': [], 'test_acc@5': []}
    print('Training...')
    for epoch in range(1, args.epochs + 1):
        encoder.train(), fc.train()
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', leave=False):
            x = x.to(device)
            y = y.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()        
            
            out = fc(encoder(x))
            loss = criterion(out, y)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        if epoch % eval_every_n_epochs == 0:
            total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0
            encoder.eval(), fc.eval()
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)

                    out = fc(encoder(x))
                    prediction = torch.argsort(out, dim=-1, descending=True)
                    total_num += y.size(0)
                    total_correct_1 += torch.sum((prediction[:, 0:1] == y.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    total_correct_5 += torch.sum((prediction[:, 0:5] == y.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                
            test_acc_1, test_acc5 = total_correct_1 / total_num * 100, total_correct_5 / total_num * 100
            test_results['test_acc@1'].append(test_acc_1)
            test_results['test_acc@5'].append(test_acc5)
            print(f"Epoch: {epoch} Test Acc@1: {test_acc_1:.2f}% Test Acc@5: {test_acc5:.2f}%")
        
    results = pd.DataFrame(test_results, index=range(eval_every_n_epochs, args.epochs+1, eval_every_n_epochs))
    results.to_csv(semisup_result_path, index_label='epoch')
