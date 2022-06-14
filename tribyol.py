import os
import argparse
import pandas as pd
import torch
import torch.optim as optim

import numpy as np
from byol import TriBYOL

import utils
from model import Model
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--lr', default=0.03, type=float, help='Learning rate')
    args = parser.parse_args()

    batch_size, epochs = args.batch_size, args.epochs
    
    model_name = 'tribyol'

    print(model_name)
    
    writer = SummaryWriter('runs/' + model_name)

    train_transform = utils.tribyol_transform
    train_data = utils.CIFAR10Triplet(root='/home/eugene/data', train=True, transform=train_transform, download=True)
    train_loader, valid_loader = utils.create_datasets(batch_size, train_data)

    result_path = f'results_tribyol_batch{batch_size}/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    results = {'train_loss': [], 'valid_loss':[]}
    
    model = Model().cuda()

    learner = TriBYOL(
        model.f,
        image_size=32,
        hidden_layer=-2,
        projection_size=128,
        projection_hidden_size=512,
        augment_fn=lambda x: x
    )

    optimizer = optim.SGD(learner.parameters(), lr=0.03, momentum=0.9, weight_decay=4e-4)
    least_loss = np.Inf
    
    for epoch in range(1, epochs + 1):
        # train
        total_loss, total_num = 0, 0
        data_bar = tqdm(train_loader)

        learner.train()
        for x1, x2, x3, _ in data_bar:
            batch_size = x1.size(0)
            x1, x2, x3 = x1.cuda(), x2.cuda(), x3.cuda()
            
            loss = learner(x1, x2, x3)
            total_loss += loss.item() * batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            learner.update_moving_average()

            total_num += batch_size
            data_bar.set_description('Epoch: [{}/{}] Train Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
        train_loss = total_loss / total_num

        # valid
        total_loss, total_num = 0, 0
        data_bar = tqdm(valid_loader)

        learner.eval()
        with torch.no_grad():
            for x1, x2, x3, _ in data_bar:
                batch_size = x1.size(0)
                x1, x2, x3 = x1.cuda(), x2.cuda(), x3.cuda()
                
                loss = learner(x1, x2, x3)

                total_num += batch_size
                total_loss += loss.item() * batch_size
                data_bar.set_description('Epoch: [{}/{}] Valid Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
        valid_loss = total_loss / total_num

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('valid_loss', valid_loss, epoch)
        results['train_loss'].append(train_loss)
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
