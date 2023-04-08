# original code from https://github.com/lucidrains/byol-pytorch

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np

from functools import wraps
import copy
import random


# helper functions
def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)

    loss = 2 - 2 * (x * y).sum(dim=-1)
    return loss

# augmentation utils
class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor
class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets
class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        if hidden.dim() > 2:
            hidden.squeeze_()
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return flatten(self.net(x))

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection = True, intermediate = False):
        if not intermediate:
            representation = self.get_representation(x)
        else:
            representation, z = self.net(x, intermediate = True)
            representation = flatten(representation)

        if not return_projection:
            return representation
        
        idx = torch.randperm(x.shape[0])
        print("idx", idx)
        print("output indexing", representation[idx])
        print("input indexing", self.get_representation(x[idx]))
        self.hidden.clear()

        projector = self._get_projector(representation)
        projection = projector(representation)

        if not intermediate:
            return projection, representation
        else:
            return projection, representation, z

# main class
class BYOL(nn.Module):
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
        super().__init__()
        self.net = net

        # default SimCLR augmentation

        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            T.RandomResizedCrop((image_size, image_size)),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size, device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
        self,
        x1,
        x2=None,
        mixup = False,
        return_embedding = False,
        return_projection = True,
    ):
        assert not (self.training and x1.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x1, return_projection = return_projection)

        if x2 is None:
            x1, x2 = self.augment1(x1), self.augment2(x1)

        online_proj_one, _ = self.online_encoder(x1)
        online_proj_two, _ = self.online_encoder(x2)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            
            target_proj_one, _ = target_encoder(x1)
            target_proj_two, _ = target_encoder(x2)

        loss_one = loss_fn(online_pred_one, target_proj_two)
        loss_two = loss_fn(online_pred_two, target_proj_one)

        byol_loss = (loss_one + loss_two).mean()

        if mixup:
            lam = np.random.beta(1.0, 1.0, size=x1.size(0))
            lam = torch.from_numpy(lam).reshape(-1, 1, 1, 1).float().to(x1.device)
            idx = torch.randperm(x1.size(0)).to(x1.device)
            mixed_x = lam * x1 + (1 - lam) * x2[idx]

            mixed_proj, _ = self.online_encoder(mixed_x)
            mixed_pred = self.online_predictor(mixed_proj)
            online_pred = lam * online_pred_one + (1 - lam) * online_pred_two[idx]

            with torch.no_grad():
                target_mixed_proj, _ = target_encoder(mixed_x)
                target_proj = lam * target_proj_one + (1 - lam) * target_proj_two[idx]

            loss_one = loss_fn(mixed_pred, target_proj)
            loss_two = loss_fn(online_pred, target_mixed_proj)
            mixup_loss = (loss_one + loss_two).mean()
        else:
            mixup_loss = 0

        loss = byol_loss + mixup_loss
        return loss, byol_loss, mixup_loss

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
            
        # prediction mixup
        loss_one = loss_fn(mixed_pred, target_proj)
        # projection mixup
        loss_two = loss_fn(online_pred, target_mixed_proj)
        loss = loss_one + loss_two
        return loss.mean()

class TriBYOL(BYOL):
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

        self.update_target_one = True
        self.target_encoder = None

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder_one = copy.deepcopy(self.online_encoder)
        target_encoder_two = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder_one, False)
        set_requires_grad(target_encoder_two, False)
        return target_encoder_one, target_encoder_two

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        if self.update_target_one:
            update_moving_average(self.target_ema_updater, self.target_encoder[0], self.online_encoder)
            self.update_target_one = False
        else:
            update_moving_average(self.target_ema_updater, self.target_encoder[1], self.online_encoder)
            self.update_target_one = True

    def forward(
        self,
        x1,
        x2=None,
        x3=None,
        return_embedding = False,
        return_projection = True,
    ): 
        assert not (self.training and x1.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x1, return_projection = return_projection)

        if x2 is not None:
            image_one, image_two, image_three = x1, x2, x3
        else:
            image_one, image_two, image_three = self.augment1(x1), self.augment2(x1), self.augment2(x1)

        online_proj_one, _ = self.online_encoder(image_one)
        online_proj_two, _ = self.online_encoder(image_two)
        online_proj_three, _ = self.online_encoder(image_three)
        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)
        online_pred_three = self.online_predictor(online_proj_three)

        with torch.no_grad():
            target_encoder_one, target_encoder_two = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_one_proj_two, _ = target_encoder_one(image_two)
            target_one_proj_one, _ = target_encoder_one(image_one)
            target_two_proj_three, _ = target_encoder_two(image_three)
            target_two_proj_one, _ = target_encoder_two(image_one)

            target_one_proj_two.detach_()
            target_one_proj_one.detach_()
            target_two_proj_three.detach_()
            target_two_proj_one.detach_()

        loss_one_two = loss_fn(online_pred_one, target_one_proj_two)
        loss_two_one = loss_fn(online_pred_two, target_one_proj_one)
        loss_one_three = loss_fn(online_pred_one, target_two_proj_three)
        loss_three_one = loss_fn(online_pred_three, target_two_proj_one)

        loss = (loss_one_two + loss_two_one) / 2 + (loss_one_three + loss_three_one) / 2
        return loss.mean()