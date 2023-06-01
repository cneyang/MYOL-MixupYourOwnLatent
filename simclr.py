import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimCLR(nn.Module):
    def __init__(
        self,
        net,
        projection_size=128,
        **kwargs
    ):
        super().__init__()
        self.net = net
        self.projector = nn.Sequential(nn.Linear(2048, 2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, projection_size)).cuda()

    def forward(
        self,
        x1,
        x2
    ):
        xs = torch.cat([x1, x2], dim=0)
        feats = self.projector(self.net(xs))

        labels = torch.cat([torch.arange(x1.size(0)) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        feats = F.normalize(feats, dim=1)

        sims = torch.mm(feats, feats.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        sims = sims[~mask].view(sims.shape[0], -1)

        positives = sims[labels.bool()].view(labels.shape[0], -1)
        negatives = sims[~labels.bool()].view(sims.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits /= 0.07

        loss = F.cross_entropy(logits, labels)

        return loss

    def update_moving_average(self):
        pass
        
