import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import SubsetRandomSampler
from byol_pytorch import BYOL


def create_datasets(batch_size, train_data):
    # trainning set 중 validation 데이터로 사용할 비율
    valid_size = 0.2

    # validation으로 사용할 trainning indices를 얻는다.
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # trainning, validation batch를 얻기 위한 sampler정의
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # load training data in batches
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=12, drop_last=True)

    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=12, drop_last=True)

    return train_loader, valid_loader

class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset."""

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target
    
    

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    # transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)

    loss = 2 - 2 * (x * y).sum(dim=-1)
    return loss

class MixupBYOL(BYOL):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        augment_fn = None,
        augment_fn2 = None,
        moving_average_decay = 0.99,
        use_momentum = True
    ):
        super().__init__(
            net,
            image_size,
            hidden_layer,
            projection_size,
            projection_hidden_size,
            augment_fn,
            augment_fn2,
            moving_average_decay,
            use_momentum
        )

    def mixup(self, mixed_x, x1, x2, lam):
        mixed_proj, _ = self.online_encoder(mixed_x)
        online_proj_one, _ = self.online_encoder(x1)
        online_proj_two, _ = self.online_encoder(x2)
        mixed_pred = self.online_predictor(mixed_proj)
        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)
        online_pred = lam * online_pred_one + (1 - lam) * online_pred_two

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_mixed_proj, _ = target_encoder(mixed_x)
            target_proj_one, _ = target_encoder(x1)
            target_proj_two, _ = target_encoder(x2)
            target_proj = lam * target_proj_one + (1 - lam) * target_proj_two
            
            target_mixed_proj = target_mixed_proj.detach()
            target_proj = target_proj.detach()

        loss_one = loss_fn(mixed_pred, target_proj)
        loss_two = loss_fn(online_pred, target_mixed_proj)
        loss = loss_one + loss_two
        return loss.mean()