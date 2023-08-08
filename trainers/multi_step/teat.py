# for paper: Exploring Memorization in Adversarial Training
# https://github.com/dongyp13/memorization-AT/blob/main/train_te.py
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.autograd import Variable
from tqdm import tqdm
from utils.helper_funcs_wandb import define_wandb_batch_metrics

from config import *


class Teat:
    def __init__(self, config: Config, device: str, lr_scheduler,
                 model: torch.nn.Module, train_loader, optimizer, targets, num_classes=10, momentum=0.9,
                 reg_weight=300, use_wandb=False, **kwargs):
        self.model = model
        self.adv_config = config.param_atk_train
        self.optimizer = optimizer
        self.device = device
        self.total_epoch = config.total_epoch
        self.train_loader = train_loader
        self.lr_scheduler = lr_scheduler

        self.beta = config.param_fixed_lam

        num_samples = len(targets)
        self.soft_labels = torch.zeros(num_samples, num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.momentum_te = momentum
        self.start_es = int(0.45 * self.total_epoch)
        self.end_es = int(0.75 * self.total_epoch)
        self.reg_weight = reg_weight

        self.bs_print = 200
        self.tqdm_bar = tqdm(total=len(self.train_loader) * self.total_epoch, file=sys.stdout, position=0, ncols=120)

        self.use_wandb = use_wandb
        if self.use_wandb:  # use wandb to record
            self.wb_metric_batch, self.wb_metric_lr, self.wb_metric_epoch, self.wb_metric_loss = define_wandb_batch_metrics()

    @staticmethod
    def sigmoid_rampup(current, start_es, end_es):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if current < start_es:
            return 0.0
        if current > end_es:
            return 1.0
        else:
            import math
            phase = 1.0 - (current - start_es) / (end_es - start_es)
            return math.exp(-5.0 * phase * phase)

    def train_epoch(self, idx_epoch):
        " idx_epoch should start from 1"
        self.model.train()

        rampup_rate = self.sigmoid_rampup(idx_epoch, self.start_es, self.end_es)
        weight = rampup_rate * self.reg_weight

        for batch_idx, (index, data, target) in enumerate(self.train_loader, 1):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            lr = self.lr_scheduler.get_last_lr()[0]

            # calculate robust loss
            loss = self._train_batch(idx_epoch, index, x_natural=data, y=target, weight=weight)
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

    def _train_batch(self, i_epoch, index, x_natural, y, weight):
        criterion_kl = nn.KLDivLoss(reduction='sum')
        self.model.eval()
        batch_size = len(x_natural)
        with torch.no_grad():
            logits = self.model(x_natural)

        if i_epoch >= self.start_es:
            prob = F.softmax(logits.detach(), dim=1)
            self.soft_labels[index] = self.momentum_te * self.soft_labels[index] + (1 - self.momentum_te) * prob
            soft_labels_batch = self.soft_labels[index] / self.soft_labels[index].sum(1, keepdim=True)

        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        for _ in range(self.adv_config.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits_adv = self.model(x_adv)
                loss_kl = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1))
                if i_epoch >= self.start_es:
                    loss = (self.beta / batch_size) * loss_kl + weight * (
                            (F.softmax(logits_adv, dim=1) - soft_labels_batch) ** 2).mean()
                else:
                    loss = loss_kl
            grad = torch.autograd.grad(loss, [x_adv])[0]

            x_adv = x_adv.detach() + self.adv_config.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural
                                        - self.adv_config.epsilon), x_natural + self.adv_config.epsilon)

            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        # compute loss
        self.model.train()
        self.optimizer.zero_grad()
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

        # calculate robust loss
        logits = self.model(x_natural)
        logits_adv = self.model(x_adv)
        loss_natural = F.cross_entropy(logits, y)
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1))
        if i_epoch >= self.start_es:
            loss = loss_natural + self.beta * loss_robust + weight * (
                    (F.softmax(logits, dim=1) - soft_labels_batch) ** 2).mean()
        else:
            loss = loss_natural + self.beta * loss_robust
        return loss
