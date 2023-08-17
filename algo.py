# original code from https://github.com/lucidrains/byol-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import wraps
import copy


# helper functions
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
    def __init__(self, input_dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
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
        representation = self.net(x)
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
        projection_size = 128,
        projection_hidden_size = 2048,
        moving_average_decay = 0.996,
        use_momentum = True
    ):
        super().__init__()
        self.net = net

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        with torch.no_grad():
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
        x2 = None,
        return_embedding = False,
        return_projection = True,
    ):
        assert not (self.training and x1.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x1, return_projection = return_projection)
        
        if x2 is None:
            x2 = x1

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

        loss = (loss_one + loss_two).mean()

        return loss
    
class MYOL(BYOL):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_size = 128,
        projection_hidden_size = 2048,
        moving_average_decay = 0.996,
        use_momentum = True
    ):
        super().__init__(
            net,
            image_size,
            hidden_layer,
            projection_size,
            projection_hidden_size,
            moving_average_decay,
            use_momentum
        )

    def forward(
        self,
        x1,
        x2 = None,
        return_embedding = False,
        return_projection = True,
        alpha = 2.0,
        gamma = 1.0
    ):
        assert not (self.training and x1.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x1, return_projection = return_projection)
        
        if x2 is None:
            x2 = x1

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

        lam = np.random.beta(alpha, alpha, size=x1.size(0))
        lam = torch.FloatTensor(lam).reshape(-1, 1, 1, 1).cuda()
        idx = torch.randperm(x1.size(0)).cuda()

        mixed_x = lam * x1 + (1 - lam) * x2[idx]
        lam = lam.reshape(-1, 1)

        mixed_proj, _ = self.online_encoder(mixed_x)
        mixed_pred = self.online_predictor(mixed_proj)

        with torch.no_grad():
            target_proj = lam * target_proj_one + (1 - lam) * target_proj_two[idx]

        mixup_loss = loss_fn(mixed_pred, target_proj).mean()

        loss = byol_loss + gamma * mixup_loss

        return loss
    
class TRIBYOL(BYOL):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_size = 128,
        projection_hidden_size = 2048,
        moving_average_decay = 0.996,
        use_momentum = True
    ):
        super().__init__(
            net,
            image_size,
            hidden_layer,
            projection_size,
            projection_hidden_size,
            moving_average_decay,
            use_momentum
        )
        self.update_target_one = True

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder_one = copy.deepcopy(self.online_encoder)
        target_encoder_two = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder_one, False)
        set_requires_grad(target_encoder_two, False)
        return target_encoder_one, target_encoder_two

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
        x2 = None,
        x3 = None,
        return_embedding = False,
        return_projection = True,
    ):
        assert not (self.training and x1.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x1, return_projection = return_projection)
        
        if x2 is None and x3 is None:
            x2 = x1
            x3 = x1

        online_proj_one, _ = self.online_encoder(x1)
        online_proj_two, _ = self.online_encoder(x2)
        online_proj_three, _ = self.online_encoder(x3)
        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)
        online_pred_three = self.online_predictor(online_proj_three)

        with torch.no_grad():
            target_encoder_one, target_encoder_two = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_one_proj_two, _ = target_encoder_one(x2)
            target_one_proj_one, _ = target_encoder_one(x1)
            target_two_proj_three, _ = target_encoder_two(x3)
            target_two_proj_one, _ = target_encoder_two(x1)

        loss_one_two = loss_fn(online_pred_one, target_one_proj_two)
        loss_two_one = loss_fn(online_pred_two, target_one_proj_one)
        loss_one_three = loss_fn(online_pred_one, target_two_proj_three)
        loss_three_one = loss_fn(online_pred_three, target_two_proj_one)

        loss = (loss_one_two + loss_two_one) / 2 + (loss_one_three + loss_three_one) / 2

        return loss.mean()
    
class IMIX(BYOL):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_size = 128,
        projection_hidden_size = 2048,
        moving_average_decay = 0.996,
        use_momentum = True
    ):
        super().__init__(
            net,
            image_size,
            hidden_layer,
            projection_size,
            projection_hidden_size,
            moving_average_decay,
            use_momentum
        )

    def mixup(self, input):
        beta = torch.distributions.beta.Beta(1.0, 1.0)
        randind = torch.randperm(input.shape[0], device=input.device)
        lam = beta.sample([input.shape[0]]).to(device=input.device)
        lam = torch.max(lam, 1. - lam)
        lam_expanded = lam.view([-1] + [1]*(input.dim()-1))
        output = lam_expanded * input + (1. - lam_expanded) * input[randind]
        return output, randind, lam

    def forward(
        self,
        x1,
        x2 = None,
        return_embedding = False,
        return_projection = True,
    ):
        assert not (self.training and x1.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x1, return_projection = return_projection)
        
        if x2 is None:
            x2 = x1

        x1, labels_aux, lam = self.mixup(x1)

        online_proj, _ = self.online_encoder(x1)
        online_pred = self.online_predictor(online_proj)
        online_pred = F.normalize(online_pred, dim=1)
        
        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj, _ = target_encoder(x2)
            target_proj = F.normalize(target_proj, dim=1)

        logits = online_pred.mm(target_proj.t())
        target_logits = lam * logits.diag() + (1. - lam) * logits[range(x2.size(0)), labels_aux]
        loss = (2. - 2. * target_logits).mean()

        return loss
    
class UNMIX(BYOL):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_size = 128,
        projection_hidden_size = 2048,
        moving_average_decay = 0.996,
        use_momentum = True
    ):
        super().__init__(
            net,
            image_size,
            hidden_layer,
            projection_size,
            projection_hidden_size,
            moving_average_decay,
            use_momentum
        )

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def forward(
        self,
        x1,
        x2 = None,
        return_embedding = False,
        return_projection = True,
    ):
        assert not (self.training and x1.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x1, return_projection = return_projection)
        
        if x2 is None:
            x2 = x1

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

        r = np.random.rand(1)
        lam = np.random.beta(1.0, 1.0)
        x_re = torch.flip(x1, (0,))

        if r < 0.5:
            mixed_x = lam * x1 + (1 - lam) * x_re
            mixed_x_re = torch.flip(mixed_x, (0,))
        else:
            mixed_x = x1.clone()
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(x1.size(), lam)
            mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x_re[:, :, bbx1:bbx2, bby1:bby2]
            mixed_x_re = torch.flip(mixed_x, (0,))
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x1.size()[-1] * x1.size()[-2]))

        mixed_proj, _ = self.online_encoder(mixed_x)
        mixed_proj_re, _ = self.online_encoder(mixed_x_re)
        mixed_p = self.online_predictor(mixed_proj)
        mixed_p_re = self.online_predictor(mixed_proj_re)

        unmix_loss = lam * loss_fn(mixed_p, target_proj_one).mean() + (1 - lam) * loss_fn(mixed_p_re, target_proj_one).mean()

        loss = byol_loss + unmix_loss

        return loss


class MOCO(nn.Module):
    def __init__(
        self,
        net,
        projection_size=128,
        moving_average_decay=0.996,
        **kwargs
    ):
        super().__init__()
        self.K = 4096
        self.m = moving_average_decay

        self.encoder_q = nn.Sequential(
            net,
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, projection_size)
        ).cuda()
        self.encoder_k = copy.deepcopy(self.encoder_q).cuda()
        
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(projection_size, self.K).cuda())
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()

            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= 0.07

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        self._dequeue_and_enqueue(k)

        loss = F.cross_entropy(logits, labels)

        return loss

    def update_moving_average(self):
        pass


class SIMCLR(nn.Module):
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
        
