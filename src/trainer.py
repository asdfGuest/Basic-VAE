import os

import torch as th
from torch.utils.data import DataLoader

from src.model import VAE


_cpu = th.device('cpu')

def mean(x:list|tuple) :
    return sum(x) / len(x)


class Trainer :
    def __init__(
            self,
            vae:VAE,
            optimizer:th.optim.Optimizer,
            dataloader:DataLoader,
            device:th.device|None=None
        ):
        self.vae = vae
        self.optimizer = optimizer
        self.dataloader = dataloader

        if device is None :
            device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.device = device


    def train(
            self,
            n_epoch:int,
            log_freq:int,
            directory:str='./checkpoint'
            ) :

        n_batch = len(self.dataloader)

        for epoch_idx in range(n_epoch) :

            recon_loss_buff = []
            regul_loss_buff = []

            for batch_idx, (x,_) in enumerate(self.dataloader) :

                x = x.to(self.device)

                mu, logstd = self.vae.encode(x)
                z = self.vae.rsample(mu, logstd)
                y = self.vae.decode(z)

                recon_loss, regul_loss = self.vae.compute_loss(x, y, mu, logstd)
                loss = recon_loss + regul_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                recon_loss_buff.append(recon_loss.item())
                regul_loss_buff.append(regul_loss.item())

                if (batch_idx+1) % log_freq == 0 :
                    print(
                        '| epoch %d/%d | batch %d/%d | recon loss %.3f | regul loss %.6f |'%(
                            epoch_idx,
                            n_epoch,
                            batch_idx,
                            n_batch,
                            mean(recon_loss_buff),
                            mean(regul_loss_buff)
                        )
                    )

                    recon_loss_buff.clear()
                    regul_loss_buff.clear()
            
            os.makedirs(directory, exist_ok=True)
            self.vae.save(directory+'/model_%d.pt'%(epoch_idx))
