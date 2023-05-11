import os
import math
import argparse
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

import dataset
from model import Model
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def adjust_learning_rate(optimizer, epoch, lr, scheduler):
    if scheduler is not None:
        scheduler.step()
    else:
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / 1000))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def warmup_learning_rate(optimizer, epoch, batch_id, total_batches, warmup_to):
    p = (batch_id + 1 + epoch * total_batches) / (10 * total_batches)
    lr = p * warmup_to

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tinyimagenet', type=str, help='Dataset')
    parser.add_argument('--algo', default='myol', type=str, help='Algorithm')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--optim', default='adam', type=str, help='Optimizer')
    parser.add_argument('--lr', default=2e-3, type=float, help='Learning rate')
    parser.add_argument('--cos', action='store_true', help='Use cosine annealing')
    parser.add_argument('--hidden_dim', default=512, type=int, help='Hidden dimension of the projection head')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    args = parser.parse_args()
    args.triplet = True if args.algo == 'tribyol' else False

    model_name = f'{args.algo}_{args.optim}{args.lr}_cos{args.cos}_{args.hidden_dim}_{args.seed}'

    result_path = f'main_result/{args.dataset}/results_{args.algo}_batch{args.batch_size}/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    print(model_name)
    if os.path.exists(result_path+f'{model_name}_{args.epochs}.pth'):
        print(model_name, 'already exists')
        import sys
        sys.exit()
        
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    writer = SummaryWriter('runs/' + f'{args.dataset}/batch{args.batch_size}/' + model_name)

    if args.dataset == 'cifar10':
        train_transform = dataset.CIFAR10.get_transform(train=True)
        train_data = dataset.CIFAR10(root='./data', train=True, transform=train_transform, download=True, triplet=args.triplet)
        image_size = 32
    elif args.dataset == 'cifar100':
        train_transform = dataset.CIFAR100.get_transform(train=True)
        train_data = dataset.CIFAR100(root='./data', train=True, transform=train_transform, download=True, triplet=args.triplet)
        image_size = 32
    elif args.dataset == 'stl10':
        train_transform = dataset.STL10.get_transform(train=True)
        train_data = dataset.STL10(root='./data', split='train+unlabeled', transform=train_transform, download=True, triplet=args.triplet)
        image_size = 64
    elif args.dataset == 'tinyimagenet':
        train_transform = dataset.TinyImageNet.get_transform(train=True)
        train_data = dataset.TinyImageNet(root='./data/tiny-imagenet-200/train', transform=train_transform, triplet=args.triplet)
        image_size = 64

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    results = {'train_loss': []}
    
    model = Model(args.dataset).cuda()

    # get class
    algo = getattr(__import__('algo'), args.algo.upper())

    learner = algo(
        model.f,
        image_size=image_size,
        hidden_layer=-2,
        projection_size=128,
        projection_hidden_size=args.hidden_dim,
        moving_average_decay=0.996,
    )

    if args.optim == 'adam':
        optimizer = optim.Adam(learner.parameters(), lr=args.lr, weight_decay=1e-6)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(learner.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    if args.cos:
        scheduler = None
        eta_min = args.lr * 0.001
        warmup_to = eta_min + (args.lr - eta_min) * (1 + math.cos(math.pi * 10 / 1000)) / 2
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[950, 975], gamma=0.2)
        warmup_to = args.lr

    scaler = torch.cuda.amp.GradScaler()
    least_loss = np.Inf
    
    for epoch in range(1, args.epochs + 1):
        # train
        total_loss, total_num = 0, 0
        data_bar = tqdm(train_loader)

        learner.train()
        adjust_learning_rate(optimizer, epoch, args.lr, scheduler)
        for i, (imgs, labels) in enumerate(data_bar):
            if epoch <= 10:
                warmup_learning_rate(optimizer, epoch, i, len(train_loader), warmup_to)

            if args.triplet:
                x1, x2, x3 = imgs
            else:
                x1, x2 = imgs

            batch_size = x1.size(0)

            with torch.cuda.amp.autocast():
                if args.triplet:
                    loss = learner(x1.cuda(), x2.cuda(), x3.cuda())
                else:
                    loss = learner(x1.cuda(), x2.cuda())

            total_num += batch_size
            total_loss += loss.item() * batch_size

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            learner.update_moving_average()

            data_bar.set_description('Epoch: [{}/{}] Train Loss: {:.4f}'.format(epoch, args.epochs, total_loss / total_num))
        
        train_loss = total_loss / total_num
        writer.add_scalar('train_loss', train_loss, epoch)
        results['train_loss'].append(train_loss)
        
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(result_path+f'{model_name}_statistics.csv', index_label='epoch')
        if epoch % 100 == 0:
            torch.save(model.state_dict(), result_path+f'{model_name}_{epoch}.pth')
