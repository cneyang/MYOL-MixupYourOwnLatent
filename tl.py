import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, STL10, MNIST, FashionMNIST, KMNIST, USPS, SVHN
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
    parser.add_argument('--dataset', default='tinyimagenet', type=str, help='Dataset')
    parser.add_argument('--target', default='cifar10', type=str, help='Target dataset')
    parser.add_argument('--algo', type=str, default='myol')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--checkpoint', type=int, default=500)
    parser.add_argument('--optim', default='sgd', type=str, help='Optimizer')
    parser.add_argument('--lr', default=0.05, type=float, help='Learning rate')
    parser.add_argument('--cos', action='store_true', help='Use cosine annealing')
    parser.add_argument('--hidden_dim', default=2048, type=int, help='Hidden dimension of the projection head')
    parser.add_argument('--finetune', action='store_true', help='Fine tune the model')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    args.epochs = 20 if args.finetune else 100

    batch_size = 256 if args.finetune else 512

    model_name = f'{args.algo}_{args.optim}{args.lr}_cos{args.cos}_{args.hidden_dim}_{args.seed}'
    model_path = f'main_result/{args.dataset}/results_{args.algo}_batch{args.batch_size}/{model_name}_{args.checkpoint}.pth'
    result_path = f'main_result/{args.dataset}/results_{args.algo}_batch{args.batch_size}/tl_{args.target}_finetune{args.finetune}_{model_name}_statistics_{args.checkpoint}.csv'
    
    print(model_name, args.checkpoint)
    if os.path.exists(result_path):
        df = pd.read_csv(result_path)
        print(f'{args.algo} {args.target} finetune{args.finetune} {df["test_acc@1"].values[-1]:.2f}')
        import sys
        sys.exit()
        
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if 'mnist' in args.target or 'usps' in args.target:
        # 3 channel
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        ])

    if args.target == 'cifar10':
        train_data = CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_data = CIFAR10(root='./data', train=False, transform=transform, download=True)
        num_class = 10
    elif args.target == 'cifar100':
        train_data = CIFAR100(root='./data', train=True, transform=transform, download=True)
        test_data = CIFAR100(root='./data', train=False, transform=transform, download=True)
        num_class = 100
    elif args.target == 'stl10':
        train_data = STL10(root='./data', split='train', transform=transform, download=True)
        test_data = STL10(root='./data', split='test', transform=transform, download=True)
        num_class = 10
    elif args.target == 'mnist':
        train_data = MNIST(root='./data', train=True, transform=transform, download=True)
        test_data = MNIST(root='./data', train=False, transform=transform, download=True)
        num_class = 10
    elif args.target == 'fashionmnist':
        train_data = FashionMNIST(root='./data', train=True, transform=transform, download=True)
        test_data = FashionMNIST(root='./data', train=False, transform=transform, download=True)
        num_class = 10
    elif args.target == 'kmnist':
        train_data = KMNIST(root='./data', train=True, transform=transform, download=True)
        test_data = KMNIST(root='./data', train=False, transform=transform, download=True)
        num_class = 10
    elif args.target == 'usps':
        train_data = USPS(root='./data', train=True, transform=transform, download=True)
        test_data = USPS(root='./data', train=False, transform=transform, download=True)
        num_class = 10
    elif args.target == 'svhn':
        train_data = SVHN(root='./data', split='train', transform=transform, download=True)
        test_data = SVHN(root='./data', split='test', transform=transform, download=True)
        num_class = 10

    train_loader = DataLoader(train_data, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=True)

    test_loader = DataLoader(test_data, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(args.dataset, pretrained_path=model_path).to(device)

    fc = FC(num_class=num_class)
    fc = fc.to(device)

    if args.finetune:
        optimizer = torch.optim.SGD([
            {'params': encoder.parameters(), 'lr': 0.01},
            {'params': fc.parameters(), 'lr': 0.1}],
            momentum=0.9,
            weight_decay=0)
    else:
        encoder.eval()
        x_train, y_train = get_features_from_encoder(encoder, train_loader, device)
        x_test, y_test = get_features_from_encoder(encoder, test_loader, device)
        
        if len(x_train.shape) > 2:
            x_train = torch.mean(x_train, dim=[2, 3])
            x_test = torch.mean(x_test, dim=[2, 3])

        train_loader, test_loader = create_data_loaders_from_arrays(x_train, y_train, x_test, y_test)

        optimizer = torch.optim.SGD(fc.parameters(), lr=0.2, momentum=0.9, weight_decay=0, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*len(train_loader), eta_min=0, last_epoch=-1)
    criterion = torch.nn.CrossEntropyLoss()
    
    eval_every_n_epochs = 1 if args.finetune else 10
    test_results = {'test_acc@1': [], 'test_acc@5': []}
    print('Training...')
    for epoch in range(1, args.epochs + 1):
        encoder.train(), fc.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            if args.finetune:
                out = fc(encoder(x))
            else:
                out = fc(x)
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

                    if args.finetune:
                        out = fc(encoder(x))
                    else:
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
    results.to_csv(result_path, index_label='epoch')
