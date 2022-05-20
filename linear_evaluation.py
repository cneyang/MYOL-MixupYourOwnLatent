import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import argparse
import pandas as pd
from tqdm import tqdm

import utils
from utils import MixupBYOL
from model import Model

class Net(nn.Module):
    def __init__(self, num_class, pretrained_path=None):
        super(Net, self).__init__()

        # encoder
        model = Model()
        if pretrained_path is not None:
            model.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
        self.f = model.f
        # learner = MixupBYOL(
        #     model.f,
        #     image_size=32,
        #     hidden_layer=-1,
        #     projection_size=128,
        #     projection_hidden_size=512,
        # )
        # learner.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
        # self.f = learner.net

        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out

# train or test for one epoch
def train_val(net, data_loader, train_optimizer=None):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0.0, 0
    data_bar = tqdm(data_loader)
    
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        
            if is_train:
                data_bar.set_description(('Train Epoch : {}/{} Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format(epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100)))
            else:
                data_bar.set_description(('Test Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'.format(total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100)))


    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--model_name', type=str, default='byol')
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()

    batch_size, epochs = args.batch_size, args.epochs
    if not args.eval_only:
        model_path = f'results_byol_batch100/{args.model_name}_100.pth'

        train_data = CIFAR10(root='/home/eugene/data', train=True, transform=utils.train_transform, download=True)

        train_loader, valid_loader = utils.create_datasets(batch_size, train_data)
        # model setup and optimizer config

        model = Net(num_class=len(train_data.classes), pretrained_path=model_path).cuda()
        for param in model.f.parameters():
            param.requires_grad = False

        optimizer = optim.Adam(model.fc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_criterion = nn.CrossEntropyLoss()
        results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [], 'valid_acc@1': []}

        best_acc = 0
        for epoch in range(1, epochs + 1):
            train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer)
            _, valid_acc_1, _ = train_val(model, valid_loader)
            results['train_loss'].append(train_loss)
            results['train_acc@1'].append(train_acc_1)
            results['train_acc@5'].append(train_acc_5)
            results['valid_acc@1'].append(valid_acc_1)
            
            if best_acc<valid_acc_1:
                best_epoch = epoch
                best_acc = valid_acc_1
                torch.save(model.state_dict(), f'results_byol_batch100/linear_{args.model_name}_model.pth')
                
                
            data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
            data_frame.to_csv(f'results_byol_batch100/linear_{args.model_name}_statistics.csv', index_label='epoch')

        print("Best epoch:", best_epoch)

    loss_criterion = nn.CrossEntropyLoss()

    test_data = CIFAR10(root='./data', train=False, transform=utils.test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    test_results = {'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    model = Net(num_class=len(test_data.classes))
    model_path = f'results_byol_batch100/linear_{args.model_name}_model.pth'
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None)
    test_results['test_loss'].append(test_loss)
    test_results['test_acc@1'].append(test_acc_1)
    test_results['test_acc@5'].append(test_acc_5)

    print(test_results)