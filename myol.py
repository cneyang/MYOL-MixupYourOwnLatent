import os
import argparse
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from byol import BYOL

import utils
from model import Model
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset')
    parser.add_argument('--batch_size', default=100, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--mixup', action='store_true', help='Use mixup')
    parser.add_argument('--n_steps', default=2, type=int, help='Number of byol steps per mixup step')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    args = parser.parse_args()

    batch_size, epochs = args.batch_size, args.epochs
    
    algo = 'myol' if args.mixup else 'byol'
    model_name = f'{algo}_{args.seed}'
    result_path = f'kcc/{args.dataset}/results_{algo}_batch{batch_size}/'
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    writer = SummaryWriter('runs/' + f'test/{args.dataset}/batch{args.batch_size}/' + model_name)

    if args.dataset == 'cifar10':
        train_transform = utils.CIFAR10Pair.get_transform(train=True)
        train_data = utils.CIFAR10Pair(root='./data', train=True, transform=train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    elif args.dataset == 'cifar100':
        train_transform = utils.CIFAR100Pair.get_transform(train=True)
        train_data = utils.CIFAR100Pair(root='./data', train=True, transform=train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    results = {'train_loss': [], 'mixup_loss': []}

    model = Model().cuda()

    learner = BYOL(
        model.f,
        image_size=32,
        hidden_layer=-2,
        projection_size=128,
        projection_hidden_size=512,
        augment_fn=lambda x: x
    )

    optimizer = optim.Adam(learner.parameters(), lr=5e-4, weight_decay=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    least_loss = np.Inf
    
    for epoch in range(1, epochs + 1):
        # train
        total_loss, total_num = 0, 0
        total_mixup_loss = 0
        data_bar = tqdm(train_loader)

        learner.train()
        for i, (x1, x2, _) in enumerate(data_bar):
            batch_size = x1.size(0)
            x1, x2 = x1.cuda(), x2.cuda()
            
            mixup = args.mixup and (i % args.n_steps == 0)
            with torch.cuda.amp.autocast():
                loss, byol_loss, mixup_loss = learner(x1, x2, mixup=mixup)

            total_num += batch_size
            total_loss += byol_loss.item() * batch_size
            total_mixup_loss += mixup_loss.item() * batch_size if mixup else 0

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            learner.update_moving_average()

            data_bar.set_description('Epoch: [{}/{}] Train Loss: {:.4f} Mixup Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num, total_mixup_loss / (total_num / args.n_steps)))
        train_loss = total_loss / total_num
        mixup_loss = total_mixup_loss / (total_num / args.n_steps)

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('mixup_loss', mixup_loss, epoch)
        results['train_loss'].append(train_loss)
        results['mixup_loss'].append(mixup_loss)
        
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(result_path+f'{model_name}_statistics.csv', index_label='epoch')

        if epoch % 10 == 0:
            torch.save(model.state_dict(), result_path+f'{model_name}_{epoch}.pth')
