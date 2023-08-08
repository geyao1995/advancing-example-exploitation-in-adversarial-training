"""
Fast is better than free: Revisiting adversarial training
https://github.com/locuslab/fast_adversarial/blob/master/CIFAR10/train_fgsm.py
"""
import sys

import torch
import wandb
from torch import nn
from tqdm import tqdm

from config import Config
from utils.helper_funcs_wandb import define_wandb_batch_metrics


class Fast:
    def __init__(self, config: Config, device: str, lr_scheduler, model: torch.nn.Module,
                 train_loader, optimizer, use_wandb=False,
                 **kwargs):
        self.model = model
        self.adv_config = config.param_atk_train
        self.optimizer = optimizer
        self.device = device

        self.total_epoch = config.total_epoch
        self.train_loader = train_loader

        self.lr_scheduler = lr_scheduler

        self.bs_print = 200
        self.tqdm_bar = tqdm(total=len(self.train_loader) * self.total_epoch, file=sys.stdout, position=0, ncols=120)

        self.use_wandb = use_wandb
        if self.use_wandb:  # use wandb to record
            self.wb_metric_batch, self.wb_metric_lr, self.wb_metric_epoch, self.wb_metric_loss = define_wandb_batch_metrics()

    def train_epoch(self, idx_epoch):
        " idx_epoch should start from 1"

        self.model.train()

        for batch_idx, (idx, data, target) in enumerate(self.train_loader, 1):
            data, target = data.to(self.device), target.to(self.device)

            lr = self.lr_scheduler.get_last_lr()[0]

            self.optimizer.zero_grad()

            # calculate robust loss
            loss = self._train_batch(x_natural=data, y=target)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            if batch_idx % self.bs_print == 0:
                self.tqdm_bar.write(f'[{idx_epoch:<2}, {batch_idx + 1:<5}] '
                                    f'Adv loss: {loss:<6.4f} '
                                    f'lr: {lr:.4f} ')

            self.tqdm_bar.update(1)
            self.tqdm_bar.set_description(f'epoch-{idx_epoch:<3} '
                                          f'batch-{batch_idx + 1:<3} '
                                          f'Adv loss-{loss:<.2f} '
                                          f'lr-{lr:.3f} ')

            if self.use_wandb:
                idx_batch_total = (idx_epoch - 1) * len(self.train_loader) + batch_idx
                wandb.log({self.wb_metric_lr: lr, self.wb_metric_batch: idx_batch_total})
                wandb.log({self.wb_metric_loss: loss.item(), self.wb_metric_batch: idx_batch_total})
                wandb.log({self.wb_metric_epoch: idx_epoch, self.wb_metric_batch: idx_batch_total})

        if idx_epoch >= self.total_epoch:
            self.tqdm_bar.clear()
            self.tqdm_bar.close()

    def _train_batch(self, x_natural, y):

        criterion = nn.CrossEntropyLoss()

        x_adv = x_natural + torch.randn_like(x_natural).uniform_(-self.adv_config.epsilon, self.adv_config.epsilon)
        x_adv = torch.clamp(x_adv, min=0, max=1).detach()
        x_adv.requires_grad = True

        logits_adv = self.model(x_adv)
        loss_adv = criterion(logits_adv, y)

        grad = torch.autograd.grad(loss_adv, x_adv,
                                   retain_graph=False, create_graph=False)[0]

        x_adv = x_adv + self.adv_config.step_size * grad.sign()
        delta = torch.clamp(x_adv - x_natural, min=-self.adv_config.epsilon, max=self.adv_config.epsilon)
        x_adv = torch.clamp(x_natural + delta, min=0, max=1).detach()

        output = self.model(x_adv)
        loss_adv = criterion(output, y)

        return loss_adv
