import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing

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
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--algo', type=str, default='byol')
    parser.add_argument('--model_name', type=str, default='byol')
    parser.add_argument('--checkpoint', type=str, default='best')
    args = parser.parse_args()

    batch_size = 512
    checkpoint = '' if args.checkpoint == 'best' else '_' + args.checkpoint

    transform = utils.test_transform
    train_data = CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_data = CIFAR10(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_data, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=True)

    test_loader = DataLoader(test_data, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = f'results_{args.algo}_batch{args.batch_size}/{args.model_name}{checkpoint}.pth'
    encoder = Encoder(pretrained_path=model_path).to(device)

    fc = FC(num_class=len(train_data.classes))
    fc = fc.to(device)

    encoder.eval()
    x_train, y_train = get_features_from_encoder(encoder, train_loader, device)
    x_test, y_test = get_features_from_encoder(encoder, test_loader, device)

    if len(x_train.shape) > 2:
        x_train = torch.mean(x_train, dim=[2, 3])
        x_test = torch.mean(x_test, dim=[2, 3])
        
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    train_loader, test_loader = create_data_loaders_from_arrays(torch.from_numpy(x_train), y_train, torch.from_numpy(x_test), y_test)

    optimizer = torch.optim.Adam(fc.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    eval_every_n_epochs = 10
    test_results = {'test_acc@1': [], 'test_acc@5': []}
    print('Training...')
    for epoch in range(1, args.epochs + 1):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()        
            
            out = fc(x)
            loss = criterion(out, y)
            
            loss.backward()
            optimizer.step()
        
        if epoch % eval_every_n_epochs == 0:
            total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

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
    results.to_csv(f'results_{args.algo}_batch{args.batch_size}/linear_{args.model_name}_statistics.csv', index_label='epoch')
