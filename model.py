import torch.nn as nn
from torchvision.models.resnet import resnet50, resnet18


def get_model(num_classes, arch='resnet50'):
    # encoder
    if arch == 'resnet18':
        model = resnet18()
        hidden_dim = 512
    elif arch == 'resnet50':
        model = resnet50()
        hidden_dim = 2048
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(hidden_dim, num_classes)
    return model
