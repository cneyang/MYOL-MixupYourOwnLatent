import os
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--algo', default='byol', type=str, help='Algorithm')
    parser.add_argument('--alpha', default=1.0, type=float, help='mixup alpha')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    args = parser.parse_args()
    args.triplet = True if args.algo == 'tribyol' else False

    model_name = f'{args.algo}_{args.seed}'

    result_path = f'{args.dataset}/results_{args.algo}_batch{args.batch_size}/'
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
        image_size = 96
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
        projection_hidden_size=512,
    )

    optimizer = optim.Adam(learner.parameters(), lr=5e-4, weight_decay=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    least_loss = np.Inf
    
    for epoch in range(1, args.epochs + 1):
        # train
        total_loss, total_num = 0, 0
        data_bar = tqdm(train_loader)

        learner.train()
        for imgs, labels in data_bar:
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
        if epoch % 10 == 0:
            torch.save(model.state_dict(), result_path+f'{model_name}_{epoch}.pth')
