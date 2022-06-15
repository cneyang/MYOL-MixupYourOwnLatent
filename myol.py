import os
import argparse
import pandas as pd
import torch
import torch.optim as optim

import numpy as np
from byol import MixupBYOL

import utils
from mixup import mixup_data
from model import Model
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=80, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--lr', default=0.03, type=float, help='Learning rate')
    parser.add_argument('--mixup_lambda', default=1, type=float, help='Lambda for mixup')
    parser.add_argument('--mixup', action='store_true', help='Use mixup')
    args = parser.parse_args()

    batch_size, epochs = args.batch_size, args.epochs
    mixup_lambda = args.mixup_lambda
    
    if args.mixup:
        model_name = 'myol_img_dim96_lambda{}'.format(int(mixup_lambda))
    else:
        model_name = 'byol'

    print(model_name)
    
    writer = SummaryWriter('runs/' + model_name)

    train_transform = utils.tribyol_transform
    train_data = utils.CIFAR10Pair(root='/home/eugene/data', train=True, transform=train_transform, download=True)
    train_loader, valid_loader = utils.create_datasets(batch_size, train_data)

    result_path = f'results_myol_batch{batch_size}/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    results = {'train_loss': [], 'mixup_loss': [], 'valid_loss':[]}
    
    model = Model().cuda()

    learner = MixupBYOL(
        model.f,
        image_size=32,
        hidden_layer=-2,
        projection_size=128,
        projection_hidden_size=512,
        augment_fn=lambda x: x
    )

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
                mixed_x, x1, x2, lam = mixup_data(x1, x2, use_cuda=True)
                mixed_x = mixed_x.detach()

                mixup_loss = learner.mixup(mixed_x, x1, x2, lam)
                mixup_loss = mixup_lambda * mixup_loss
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
        if epoch % (epochs//5) == 0:
            torch.save(model.state_dict(), result_path+f'{model_name}_{epoch}.pth')

    print("Best epoch: ", best_epoch)
