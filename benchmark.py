import os
import math
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

import dataset
from model import Model
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tinyimagenet', type=str, help='Dataset')
    parser.add_argument('--algo', default='myol', type=str, help='Algorithm')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--optim', default='sgd', type=str, help='Optimizer')
    parser.add_argument('--cos', default=False, type=bool, help='Use cosine learning rate')
    parser.add_argument('--lr', default=0.05, type=float, help='Learning rate')
    parser.add_argument('--hidden_dim', default=2048, type=int, help='Hidden dimension of the projection head')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    args = parser.parse_args()
    args.triplet = True if args.algo == 'tribyol' else False

    model_name = f'{args.algo}_{args.optim}{args.lr}_cos{args.cos}_{args.hidden_dim}_{args.seed}'

    result_path = f'main_result/{args.dataset}/results_{args.algo}_batch{args.batch_size}/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    print(model_name)
        
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

    optimizer = optim.SGD(learner.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)


    scaler = torch.cuda.amp.GradScaler()
    least_loss = np.Inf
    
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    total_loss, total_num = 0, 0
    times = []

    learner.train()
    for i, (imgs, labels) in enumerate(train_loader):

        if args.triplet:
            x1, x2, x3 = imgs
        else:
            x1, x2 = imgs

        batch_size = x1.size(0)

        torch.cuda.synchronize()

        start_event.record()

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

        end_event.record()

        torch.cuda.synchronize()

        gpu_time = start_event.elapsed_time(end_event)
        times.append(gpu_time)

        if i == 10:
            print(f"GPU time: {np.mean(times):.2f}", "milliseconds")

            max_memory_cached = torch.cuda.max_memory_cached()

            print(f"Max memory: {max_memory_cached / 1024**2:.2f}", "MB")
            print()
            break