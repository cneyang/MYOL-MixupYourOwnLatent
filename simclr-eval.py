import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, STL10

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import utils
from model import Model


class Encoder(nn.Module):
    def __init__(self, pretrained_path=None):
        super(Encoder, self).__init__()

        # encoder
        model = Model()
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--algo', type=str, default='byol')
    parser.add_argument('--model_name', type=str, default='byol')
    parser.add_argument('--checkpoint', type=str, default='best')
    parser.add_argument('--seed', type=int, default=27407)
    args = parser.parse_args()

    checkpoint = '' if args.checkpoint == 'best' else '_' + args.checkpoint

    print(args.model_name, args.checkpoint)
    if os.path.exists(f'kcc/{args.dataset}/results_{args.algo}_batch{args.batch_size}/linear_{args.model_name}_{args.seed}_statistics{checkpoint}.csv'):
        print('Already done')
        import sys
        sys.exit()
        
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.dataset == 'cifar10':
        transform = utils.CIFAR10Pair.get_transform(train=False)
        train_data = CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_data = CIFAR10(root='./data', train=False, transform=transform, download=True)
        num_class = 10
    elif args.dataset == 'cifar100':
        transform = utils.CIFAR100Pair.get_transform(train=False)
        train_data = CIFAR100(root='./data', train=True, transform=transform, download=True)
        test_data = CIFAR100(root='./data', train=False, transform=transform, download=True)
        num_class = 100
    elif args.dataset == 'stl10':
        transform = utils.STL10Pair.get_transform(train=False)
        train_data = STL10(root='./data', split='train', transform=transform, download=True)
        test_data = STL10(root='./data', split='test', transform=transform, download=True)
        num_class = 10

    train_loader = DataLoader(train_data, batch_size=512,
                            num_workers=0, drop_last=False, shuffle=True)

    test_loader = DataLoader(test_data, batch_size=512,
                            num_workers=0, drop_last=False, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = f'kcc/{args.dataset}/results_{args.algo}_batch{args.batch_size}/{args.model_name}_{args.seed}{checkpoint}.pth'
    encoder = Encoder(pretrained_path=model_path).to(device)
    encoder.requires_grad_(False)

    fc = FC(num_class=num_class)
    fc.linear.weight.data.normal_(0, 0.01)
    fc.linear.bias.data.zero_()
    fc = fc.to(device)

    optimizer = torch.optim.SGD(fc.parameters(), lr=0.2, momentum=0.9, weight_decay=0, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*len(train_loader), eta_min=0, last_epoch=-1)
    criterion = torch.nn.CrossEntropyLoss()
    
    eval_every_n_epochs = 10
    test_results = {'test_acc@1': [], 'test_acc@5': []}
    print('Training...')
    for epoch in range(1, args.epochs + 1):
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', leave=False):
            x = x.to(device)
            y = y.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()        
            
            x = encoder(x)
            out = fc(x)
            loss = criterion(out, y)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        if epoch % eval_every_n_epochs == 0:
            total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                x = encoder(x)
                out = fc(x)
                prediction = torch.argsort(out, dim=-1, descending=True)
                total_num += y.size(0)
                total_correct_1 += torch.sum((prediction[:, 0:1] == y.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                total_correct_5 += torch.sum((prediction[:, 0:5] == y.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                
            test_acc_1, test_acc5 = total_correct_1 / total_num * 100, total_correct_5 / total_num * 100
            test_results['test_acc@1'].append(test_acc_1)
            test_results['test_acc@5'].append(test_acc5)
            print(f"Epoch: {epoch} Test Acc@1: {test_acc_1:.2f}% Test Acc@5: {test_acc5:.2f}%")
        
    results = pd.DataFrame(test_results, index=range(eval_every_n_epochs, args.epochs+1, eval_every_n_epochs))
    results.to_csv(f'kcc/{args.dataset}/results_{args.algo}_batch{args.batch_size}/linear_{args.model_name}_{args.seed}_statistics{checkpoint}.csv', index_label='epoch')
