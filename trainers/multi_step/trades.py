# from: https://github.com/yaodongyu/TRADES/blob/master/trades.py
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.autograd import Variable
from tqdm import tqdm
from utils.helper_funcs_wandb import define_wandb_batch_metrics

from config import *


class Trades:
    def __init__(self, config: Config, device: str, lr_scheduler,
                 model: torch.nn.Module, train_loader, optimizer, use_wandb=False, **kwargs):

        self.model = model
        self.adv_config = config.param_atk_train
        self.optimizer = optimizer
        self.device = device
        self.total_epoch = config.total_epoch
        self.train_loader = train_loader
        self.lr_scheduler = lr_scheduler

        self.beta = config.param_fixed_lam

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

            self.optimizer.zero_grad()
            lr = self.lr_scheduler.get_last_lr()[0]

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
            self.tqdm_bar.close()

    def _train_batch(self, x_natural, y):
        # define KL-loss
        criterion_kl = nn.KLDivLoss(reduction='sum')
        self.model.eval()
        batch_size = len(x_natural)
        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

        with torch.no_grad():
            logits_nat = self.model(x_natural)

        for _ in range(self.adv_config.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(self.model(x_adv), dim=1),
                                       F.softmax(logits_nat, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + self.adv_config.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - self.adv_config.epsilon),
                              x_natural + self.adv_config.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        self.model.train()

        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        # zero gradient
        self.optimizer.zero_grad()
        # calculate robust loss
        logits = self.model(x_natural)
        loss_natural = F.cross_entropy(logits, y)
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(self.model(x_adv), dim=1),
                                                        F.softmax(logits, dim=1))
        loss = loss_natural + self.beta * loss_robust
        return loss
