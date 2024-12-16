import torch as th
import torch.nn as nn

from typing import List, Tuple, Type


CONVNET_ARCH_TYPE = List[Tuple[int,...]]
FCNET_ARCH_TYPE = List[int]


class Encoder(nn.Module) :
    def __init__(
                self,
                cnn_arch:CONVNET_ARCH_TYPE,
                fc_arch:FCNET_ARCH_TYPE,
                act_fn:Type[nn.Module]
            ) :
        super(Encoder, self).__init__()

        conv_net = []
        for arch in cnn_arch :
            conv_net.append(nn.Conv2d(*arch))
            conv_net.append(act_fn())
        conv_net.append(nn.Flatten())
        self.conv_net = nn.Sequential(*conv_net)

        mu_net = []
        for k in range(len(fc_arch)-1) :
            mu_net.append(nn.Linear(fc_arch[k],fc_arch[k+1]))
            mu_net.append(act_fn())
        mu_net.pop()
        self.mu_net = nn.Sequential(*mu_net)

        logstd_net = []
        for k in range(len(fc_arch)-1) :
            logstd_net.append(nn.Linear(fc_arch[k],fc_arch[k+1]))
            logstd_net.append(act_fn())
        logstd_net.pop()
        self.logstd_net = nn.Sequential(*logstd_net)

    def forward(self, x) :
        x = self.conv_net(x)
        return self.mu_net(x), self.logstd_net(x)


class Decoder(nn.Module) :
    def __init__(
                self,
                fc_arch:FCNET_ARCH_TYPE,
                cnn_arch:CONVNET_ARCH_TYPE,
                cnn_in_shape:Tuple[int,...],
                act_fn:Type[nn.Module]
            ) :
        super(Decoder, self).__init__()

        fc_net = []
        for k in range(len(fc_arch)-1) :
            fc_net.append(nn.Linear(fc_arch[k],fc_arch[k+1]))
            fc_net.append(act_fn())
        fc_net.append(nn.Unflatten(dim=1,unflattened_size=cnn_in_shape))
        self.fc_net = nn.Sequential(*fc_net)

        conv_net = []
        for arch in cnn_arch :
            conv_net.append(nn.ConvTranspose2d(*arch))
            conv_net.append(act_fn())
        conv_net.pop()
        self.conv_net = nn.Sequential(*conv_net)

    def forward(self, z) :
        return self.conv_net(self.fc_net(z))


class VAE(nn.Module) :
    def __init__(
            self,
            en_cnn_arch:CONVNET_ARCH_TYPE,
            en_fc_arch:FCNET_ARCH_TYPE,
            de_fc_arch:FCNET_ARCH_TYPE,
            de_cnn_arch:CONVNET_ARCH_TYPE,
            de_cnn_in_shape:Tuple[int,...],
            activ_fn:Type[nn.Module]
        ) :
        super(VAE, self).__init__()
        self.encoder = Encoder(en_cnn_arch, en_fc_arch, activ_fn)
        self.decoder = Decoder(de_fc_arch, de_cnn_arch, de_cnn_in_shape, activ_fn)

    def encode(self, x:th.Tensor) -> Tuple[th.Tensor,th.Tensor]:
        return self.encoder(x)

    def decode(self, z:th.Tensor) -> th.Tensor:
        return self.decoder(z)

    def rsample(self, mu:th.Tensor, logstd:th.Tensor) -> th.Tensor:
        std = th.exp(logstd)
        eps = th.randn_like(mu)
        return mu + std * eps

    def compute_loss(self, x:th.Tensor, y:th.Tensor, mu:th.Tensor, logstd:th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        logvar = logstd * 2
        recon_loss = th.sum(th.pow(x-y,2), dim=(1,2,3)).mean()
        regul_loss = th.sum(0.5 * (logvar.exp() + th.pow(mu,2) - logvar - 1), dim=1).mean()
        return recon_loss, regul_loss

    def save(self, path:str) :
        th.save(self.state_dict(), path)

    def load(self, path:str) :
        self.load_state_dict(th.load(path))
