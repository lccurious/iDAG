# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import numpy as np

from domainbed.lib import wide_resnet


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SqueezeLastTwo(nn.Module):
    """
    A module which squeezes the last two dimensions,
    ordinary squeeze can be a problem for batch size 1
    """

    def __init__(self):
        super(SqueezeLastTwo, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], x.shape[1])


class MLP(nn.Module):
    """Just  an MLP"""

    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams["mlp_width"])
        self.dropout = nn.Dropout(hparams["mlp_dropout"])
        self.hiddens = nn.ModuleList(
            [
                nn.Linear(hparams["mlp_width"], hparams["mlp_width"])
                for _ in range(hparams["mlp_depth"] - 2)
            ]
        )
        self.output = nn.Linear(hparams["mlp_width"], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class CMNIST_MLP(nn.Module):
    "For fairly comparison with IRM & BIRM"
    def __init__(self, input_shape, hparams):
        super(CMNIST_MLP, self).__init__()
        self.input_shape = input_shape
        self.hparams = hparams
        self.hidden_dim = hparams["hidden_dim"]
        self.grayscale_model = hparams["grayscale_model"]
        if self.grayscale_model:
          lin1 = nn.Linear(14 * 14, self.hidden_dim)
        else:
          lin1 = nn.Linear(2 * 14 * 14, self.hidden_dim)
        lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        lin3 = nn.Linear(self.hidden_dim, 1)
        for lin in [lin1, lin2, lin3]:
          nn.init.xavier_uniform_(lin.weight)
          nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True)) #, lin3)
        self.n_outputs = self.hidden_dim

    def forward(self, input):
        if self.grayscale_model:
          out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
        else:
          out = input.view(input.shape[0], 2 * 14 * 14)
        out = self._main(out)
        return out


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""

    def __init__(self, input_shape, hparams, network=None):
        super(ResNet, self).__init__()
        if hparams["resnet18"]:
            if network is None:
                network = torchvision.models.resnet18(pretrained=hparams["pretrained"])
            self.network = network
            self.n_outputs = 512
        else:
            if network is None:
                network = torchvision.models.resnet50(pretrained=hparams["pretrained"])
            self.network = network
            self.n_outputs = 2048

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams["resnet_dropout"])
        self.freeze_bn()

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        if self.hparams["freeze_bn"] is False:
            return

        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """

    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.squeezeLastTwo = SqueezeLastTwo()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = self.squeezeLastTwo(x)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], 128, hparams)
    elif input_shape[1:3] == (14, 14):
        return CMNIST_MLP(input_shape, hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.0)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError(f"Input shape {input_shape} is not supported")


class LocallyConnected(nn.Module):
    """
    Local linear layer, i.e., Conv1dLocal() with filter size 1.

    Args:
        num_linear: num of local linear layers
        in_features: m1
        out_features: m2
        bias: whether to include bias

    Shape:
        - Input: [n, d, m1]
        - Output: [n, d, m2]

    Attributes:
        weight: [d, m1, m2]
        bias: [d, m2]
    """
    def __init__(self, num_linear, in_features, out_features, bias=True):
        super(LocallyConnected, self).__init__()
        self.num_linear = num_linear
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(num_linear,
                                                in_features,
                                                out_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        k = 1.0 / self.in_features
        bound = np.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor):
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        out = torch.matmul(inputs.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(dim=2)
        if self.bias is not None:
            # [n, d, m2] += [d, m2]
            out += self.bias
        return out

    def extra_repr(self):
        return 'num_linear={}, in_features={}, out_features={}, bias={}'.format(
            self.num_linear, self.in_features, self.out_features,
            self.bias is not None
        )


class TraceExpm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # detach so we can cast to NumPy
        E = torch.matrix_exp(input.detach().numpy())
        f = np.trace(E)
        E = torch.from_numpy(E)
        ctx.save_for_backward(E)
        return torch.as_tensor(f, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        E, = ctx.saved_tensors
        grad_input = grad_output * E.t()
        return grad_input


class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True):
        super(NotearsMLP, self).__init__()
        assert len(dims) >=2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        self.d = dims[0]
        self.register_buffer("_identity", torch.eye(d))
        # fc1: variable spliting for l1 ref: <http://arxiv.org/abs/1909.13189>
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        # specific bounds for customize optimizer
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(nn.Sigmoid())
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.Sequential(*layers)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):
        # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)
        x = x.view(-1, self.dims[0], self.dims[1])
        x = self.fc2(x)
        x = x.squeeze(dim=2)
        return x

    def h_func(self):
        """
        Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG
        """
        d = self.dims[0]
        # [j * m1, i]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight
        fc1_weight = fc1_weight.view(d, -1, d)
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()
        # h = torch.trace(torch.matrix_exp(A)) - d
        # A different formulation, slightly faster at the cost of numerical stability
        M = self._identity + A / self.d
        E = torch.matrix_power(M, self.d - 1)
        h = (E.t() * M).sum() - self.d
        return h

    def l2_reg(self):
        """
        Take 2-norm-squared of all parameters
        """
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight
        reg += torch.sum(fc1_weight ** 2)

        for fc in self.fc2:
            if hasattr(fc, 'weight'):
                reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """
        Take l1 norm of fc1 weight
        """
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:
        """
        Get W from fc1 weight, take 2-norm over m1 dim
        """
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight
        fc1_weight = fc1_weight.view(d, -1, d)
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()
        W = torch.sqrt(A)
        W = W.cpu().detach().numpy()
        return W

    @torch.no_grad()
    def fc1_to_p_sub(self) -> torch.Tensor:
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight
        fc1_weight = fc1_weight.view(d, -1, d)
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()
        P_sub = torch.inverse(self._identity - A)
        return P_sub


class LinearNotears(nn.Module):
    def __init__(self, dims, loss_type='l2'):
        super(LinearNotears, self).__init__()
        self.dims = dims
        self.loss_type = loss_type
        self.register_buffer("_I", torch.eye(dims))
        self.weight_pos = nn.Parameter(torch.zeros(dims, dims))
        self.weight_neg = nn.Parameter(torch.zeros(dims, dims))

    def _adj(self):
        return self.weight_pos - self.weight_neg

    def h_func(self):
        W = self._adj()
        E = torch.matrix_exp(W * W)
        h = torch.trace(E) - self.dims
        # G_h = E.T * W * 2
        return h

    def _h_faster(self):
        W = self._adj()
        M = self._I + W * W / self.dims
        E = torch.matrix_power(M, self.dims - 1)
        h = (E.T * M).sum() - self.dims
        return h

    def w_l1_reg(self):
        reg = torch.sum(self.weight_pos + self.weight_neg)
        return reg

    def forward(self, x):
        W = self._adj()
        M = x @ W
        return M

    def w_to_p_sub(self):
        W = self._adj()
        P_sub = torch.inverse(self._I - W)
        return P_sub


class NotearsClassifier(nn.Module):
    def __init__(self, dims, num_classes):
        super(NotearsClassifier, self).__init__()
        self.dims = dims
        self.num_classes = num_classes
        self.weight_pos = nn.Parameter(torch.zeros(dims + 1, dims + 1))
        self.weight_neg = nn.Parameter(torch.zeros(dims + 1, dims + 1))
        # self.weight_pos = nn.Parameter(torch.triu(torch.rand(dims + 1, dims + 1), diagonal=1))
        # self.weight_neg = nn.Parameter(torch.triu(torch.rand(dims + 1, dims + 1), diagonal=1))
        # nn.init.kaiming_normal_(self.weight_pos, mode='fan_in')
        # nn.init.kaiming_normal_(self.weight_neg, mode='fan_in')
        self.register_buffer("_I", torch.eye(dims + 1))
        self.register_buffer("_repeats", torch.ones(dims + 1).long())
        self._repeats[-1] *= num_classes

    def _adj(self):
        return self.weight_pos - self.weight_neg

    def _adj_sub(self):
        W = self._adj()
        return torch.matrix_exp(W * W)

    def h_func(self):
        W = self._adj()
        E = torch.matrix_exp(W * W)
        h = torch.trace(E) - self.dims - 1
        return h

    def w_l1_reg(self):
        reg = torch.sum(self.weight_pos + self.weight_neg)
        return reg

    def forward(self, x, y=None):
        W = self._adj()
        W_sub = self._adj_sub()
        if y is not None:
            # one_hot = F.one_hot(y)
            # x_aug = torch.cat((x, one_hot), dim=1)
            # x: n_outputs + num_classes
            # W_aug = torch.repeat_interleave(W, self._repeats, dim=0)
            x_aug = torch.cat((x, y.unsqueeze(1), dim=1))
            M = x_aug @ W_aug
            # masked_x = x * W[:self.dims, -1].unsqueeze(0)
            masked_x = x * W_sub[:self.dims, -1].unsqueeze(0)
            # reconstruct variables, classification logits
            return M[:, :self.dims], masked_x
        else:
            # masked_x = x * W[:self.dims, -1].unsqueeze(0).detach()
            masked_x = x * W_sub[:self.dims, -1].unsqueeze(0).detach()
            return masked_x

    def mask_feature(self, x):
        W_sub = self._adj_sub()
        mask = W_sub[:self.dims, -1].unsqueeze(0).detach()
        return x * mask

    @torch.no_grad()
    def projection(self):
        self.weight_pos.data.clamp_(0, None)
        self.weight_neg.data.clamp_(0, None)
        self.weight_pos.data.fill_diagonal_(0)
        self.weight_neg.data.fill_diagonal_(0)

    @torch.no_grad()
    def masked_ratio(self):
        W = self._adj()
        return torch.norm(W[:self.dims, -1], p=0)


def encoder(hparams):
    if hparams["resnet18"] == False:
        n_outputs = 2048
    else:
        n_outputs = 512
    if hparams['dataset'] == "OfficeHome":
        scale_weights = 12
        pcl_weights = 1
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 512
        encoder = nn.Sequential(
            nn.Linear(n_outputs, hparams['hidden_size']),
            nn.BatchNorm1d(hparams['hidden_size']),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(hparams['hidden_size'], hparams['out_dim']),
        )
    elif hparams['dataset'] == "PACS":
        scale_weights = 12
        pcl_weights = 1
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 256
        encoder = nn.Sequential(
            nn.Linear(n_outputs, hparams['hidden_size']),
            nn.BatchNorm1d(hparams['hidden_size']),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(hparams['hidden_size'], hparams['out_dim']),
        )

    elif hparams['dataset'] == "TerraIncognita":
        scale_weights = 12
        pcl_weights = 1
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 512
        encoder = nn.Sequential(
            nn.Linear(n_outputs, hparams['hidden_size']),
            nn.BatchNorm1d(hparams['hidden_size']),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(hparams['hidden_size'], hparams['hidden_size']),
            nn.BatchNorm1d(hparams['hidden_size']),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(hparams['hidden_size'], hparams['out_dim']),
        )
    else:
        pass

    return encoder, scale_weights, pcl_weights


def fea_proj(hparams):
    if hparams['dataset'] == "OfficeHome":
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 512
        fea_proj = nn.Sequential(
            nn.Linear(hparams['out_dim'],
                      hparams['hidden_size']),
            dropout,
            nn.Linear(hparams['hidden_size'],
                      hparams['out_dim']),
        )
        fc_proj = nn.Parameter(
            torch.FloatTensor(hparams['out_dim'],
                              hparams['out_dim'])
        )
    elif hparams['dataset'] == "PACS":
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 256
        hparams['out_dim'] = 256
        fea_proj = nn.Sequential(
            nn.Linear(hparams['out_dim'],
                      hparams['out_dim']),
        )
        fc_proj = nn.Parameter(
            torch.FloatTensor(hparams['out_dim'],
                              hparams['out_dim'])
        )

    elif hparams['dataset'] == "TerraIncognita":
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 512
        fea_proj = nn.Sequential(
            nn.Linear(hparams['out_dim'],
                      hparams['out_dim']),
        )
        fc_proj = nn.Parameter(
            torch.FloatTensor(hparams['out_dim'],
                              hparams['out_dim'])
        )
    else:
        pass

    return fea_proj, fc_proj

