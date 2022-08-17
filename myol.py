import os
import argparse
from matplotlib.transforms import Transform
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from byol import MixupBYOL

import utils
from mixup import mixup_data
from model import Model
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset')
    parser.add_argument('--batch_size', default=100, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=80, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--optim', default='sgd', type=str, help='Optimizer')
    parser.add_argument('--lr', default=0.03, type=float, help='Learning rate')
    parser.add_argument('--alpha', default=1.0, type=float, help='mixup alpha')
    parser.add_argument('--beta', default=1.0, type=float, help='mixup weight')
    parser.add_argument('--mixup', action='store_true', help='Use mixup')
    parser.add_argument('--nmix', action='store_true', help='Use mixup')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size, epochs = args.batch_size, args.epochs
    
    if args.mixup:
        if args.nmix:
            model_name = 'myol_nmix_{}_alpha{}_{}'.format(args.optim, args.alpha, args.seed)
        else:
            model_name = 'myol_{}_alpha{}_{}'.format(args.optim, args.alpha, args.seed)
    else:
        model_name = 'byol_{}_{}'.format(args.optim, args.seed)

    algo = 'myol' if args.mixup else 'byol'
    result_path = f'{args.dataset}/results_{algo}_batch{batch_size}/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    print(model_name)
    if os.path.exists(result_path+f'{model_name}_{args.epochs}.pth'):
        print(model_name, 'already exists')
        import sys
        sys.exit()
    
    writer = SummaryWriter('runs/' + f'{args.dataset}/batch{args.batch_size}/' + model_name)

    if args.dataset == 'cifar10':
        train_transform = utils.tribyol_transform
        train_data = utils.CIFAR10Pair(root='./data', train=True, transform=train_transform, download=True)
        train_loader, valid_loader = utils.create_datasets(batch_size, train_data)
    # elif args.dataset == 'flowers102':
    #     train_transform = utils.tribyol_transform
    #     train_data = utils.Flowers102Pair(root='./data', split='train', transform=train_transform, download=True)
    #     valid_data = utils.Flowers102Pair(root='./data', split='val', transform=train_transform, download=True)
    #     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    #     valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=4)
    # elif args.dataset == 'aircraft':
    #     train_transform = utils.tribyol_transform
    #     train_data = utils.AircraftPair(root='./data', split='train', transform=train_transform, download=True)
    #     valid_data = utils.AircraftPair(root='./data', split='val', transform=train_transform, download=True)
    #     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    #     valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=4)


    results = {'train_loss': [], 'mixup_loss': [], 'valid_loss':[]}
    
    model = Model().cuda()

    learner = MixupBYOL(
        model.f,
        image_size=96,
        hidden_layer=-2,
        projection_size=128,
        projection_hidden_size=512,
        augment_fn=lambda x: x
    )

    if args.optim == 'adam':
        optimizer = optim.Adam(learner.parameters(), lr=5e-4, weight_decay=1e-6)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(learner.parameters(), lr=args.lr, momentum=0.9, weight_decay=4e-4)
    least_loss = np.Inf
    
    for epoch in range(1, epochs + 1):
        # train
        total_loss, total_num = 0, 0
        data_bar = tqdm(train_loader)

        learner.train()
        for x1, x2, _ in data_bar:
            batch_size = x1.size(0)
            x1, x2 = x1.cuda(), x2.cuda()
            
            loss = learner(x1, x2)
            total_loss += loss.item() * batch_size

            if args.mixup:
                mixed_x, x1, x2, lam = mixup_data(x1, x2, alpha=args.alpha, use_cuda=True)
                mixed_x = mixed_x.detach()

                mixup_loss = learner.mixup(mixed_x, x1, x2, lam) * args.beta
                loss += mixup_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            learner.update_moving_average()

            total_num += batch_size
            total_mixup_loss = mixup_loss.item() * batch_size if args.mixup else 0
            data_bar.set_description('Epoch: [{}/{}] Train Loss: {:.4f} Mixup Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num, total_mixup_loss / total_num))
        train_loss = total_loss / total_num
        mixup_loss = total_mixup_loss / total_num

        # valid
        total_loss, total_num = 0, 0
        data_bar = tqdm(valid_loader)

        learner.eval()
        with torch.no_grad():
            for x1, x2, _ in data_bar:
                batch_size = x1.size(0)
                x1, x2 = x1.cuda(), x2.cuda()
                
                loss = learner(x1, x2)

                total_num += batch_size
                total_loss += loss.item() * batch_size
                data_bar.set_description('Epoch: [{}/{}] Valid Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
        valid_loss = total_loss / total_num

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('mixup_loss', mixup_loss, epoch)
        writer.add_scalar('valid_loss', valid_loss, epoch)
        results['train_loss'].append(train_loss)
        results['mixup_loss'].append(mixup_loss)
        results['valid_loss'].append(valid_loss)
        
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(result_path+f'{model_name}_statistics.csv', index_label='epoch')
        if valid_loss < least_loss:
            least_loss = valid_loss
            best_epoch = epoch
            torch.save(model.state_dict(), result_path+f'{model_name}.pth')
        if epoch % 10 == 0:
            torch.save(model.state_dict(), result_path+f'{model_name}_{epoch}.pth')

    print("Best epoch: ", best_epoch)
