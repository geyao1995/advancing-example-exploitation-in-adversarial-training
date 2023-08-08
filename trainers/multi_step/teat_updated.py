# paper: Exploring Memorization in Adversarial Training
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


class TeatUpdated:
    def __init__(self, config: Config, device: str, lr_scheduler,
                 model: torch.nn.Module, train_loader, optimizer, num_classes: int, targets,
                 momentum_te: float = 0.9, reg_weight: int = 300, use_wandb=False,
                 **kwargs):

        self.model = model
        self.adv_config = config.param_atk_train
        self.optimizer = optimizer
        self.device = device
        self.total_epoch = config.total_epoch
        self.train_loader = train_loader
        self.lr_scheduler = lr_scheduler
        num_samples = len(targets)
        self.num_samples = num_samples
        self.num_cls = num_classes

        # params for TE
        self.soft_labels = torch.zeros(num_samples, num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.momentum_te = momentum_te
        self.epoch_start_te = int(0.45 * self.total_epoch)
        self.epoch_end_te = int(0.75 * self.total_epoch)
        self.reg_weight = reg_weight

        self.probs_global = torch.zeros((num_samples,), dtype=torch.float)
        self.logits_current_epoch = torch.zeros((num_samples, num_classes))
        self.targets = torch.from_numpy(targets)

        # parse rcat parameters
        self.epoch_start_collect = 3
        self.mometum_factor = 0.9

        self.lam_min = config.param_lam_range.lam_min
        self.lam_max = config.param_lam_range.lam_max

        self.bs_print = 200
        self.tqdm_bar = tqdm(total=len(self.train_loader) * self.total_epoch,
                             file=sys.stdout, position=0, ncols=120)

        self.use_wandb = use_wandb
        if self.use_wandb:  # use wandb to record
            self.wb_metric_batch, self.wb_metric_lr, self.wb_metric_epoch, self.wb_metric_loss = define_wandb_batch_metrics()

    def _sigmoid_rampup(self, current):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if current < self.epoch_start_te:
            return 0.0
        if current > self.epoch_end_te:
            return 1.0
        else:
            import math
            phase = 1.0 - (current - self.epoch_start_te) / (self.epoch_end_te - self.epoch_start_te)
            return math.exp(-5.0 * phase * phase)

    def _assign_lam_to_samples(self):
        ids_sorted = torch.argsort(self.probs_global)  # gap rise
        lams_for_samples = torch.empty((self.num_samples,), dtype=torch.float)
        lams = torch.linspace(self.lam_min, self.lam_max, steps=self.num_samples)
        lams_for_samples[ids_sorted] = lams

        return lams_for_samples

    def _collect_global_probs(self):
        probs_all = F.softmax(self.logits_current_epoch, dim=-1)
        probs = probs_all[range(self.num_samples), self.targets]
        self.probs_global = self.mometum_factor * self.probs_global + (1 - self.mometum_factor) * probs

    def train_epoch(self, idx_epoch):
        """
        idx_epoch should start from 1
        """

        if idx_epoch >= self.epoch_start_collect:
            self._collect_global_probs()
            lams_rob_for_sample = self._assign_lam_to_samples()
        else:
            lams_rob_for_sample = torch.full((self.num_samples,), self.lam_min, dtype=torch.float)

        rampup_rate = self._sigmoid_rampup(idx_epoch)
        reg_weight = rampup_rate * self.reg_weight

        self.model.train()

        for batch_idx, (idx, data, target) in enumerate(self.train_loader, 1):

            self.optimizer.zero_grad()
            lr = self.lr_scheduler.get_last_lr()[0]

            lams = lams_rob_for_sample[idx]

            # calculate robust loss
            loss, logits_adv = self._train_batch(
                i_epoch=idx_epoch, index=idx,
                x_natural=data.to(self.device), y=target.to(self.device),
                optimizer=self.optimizer,
                step_size=self.adv_config.step_size,
                epsilon=self.adv_config.epsilon,
                perturb_steps=self.adv_config.perturb_steps,
                lams=lams, reg_weight=reg_weight)

            if idx_epoch >= self.epoch_start_collect - 1:
                self.logits_current_epoch[idx] = logits_adv

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
                                          f'lr-{lr:.3f} '
                                          f'b-rob-{torch.mean(lams):.3f}')

            if self.use_wandb:
                idx_batch_total = (idx_epoch - 1) * len(self.train_loader) + batch_idx
                wandb.log({self.wb_metric_lr: lr, self.wb_metric_batch: idx_batch_total})
                wandb.log({self.wb_metric_loss: loss.item(), self.wb_metric_batch: idx_batch_total})
                wandb.log({self.wb_metric_epoch: idx_epoch, self.wb_metric_batch: idx_batch_total})

        if idx_epoch >= self.total_epoch:
            self.tqdm_bar.close()

    def _train_batch(self, i_epoch, index, x_natural, y, optimizer, step_size=0.007, epsilon=0.031,
                     perturb_steps=10, lams=None, reg_weight=None):
        batch_size = len(x_natural)
        # define KL-loss
        criterion_kl = nn.KLDivLoss(reduction='none')
        self.model.eval()
        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

        with torch.no_grad():
            logits_nat = self.model(x_natural)

        if i_epoch >= self.epoch_start_te:
            prob = F.softmax(logits_nat.detach(), dim=1)
            self.soft_labels[index] = self.momentum_te * self.soft_labels[index] + (1 - self.momentum_te) * prob
            soft_labels_batch = self.soft_labels[index] / self.soft_labels[index].sum(1, keepdim=True)

        for _ in range(perturb_steps):
            x_adv.requires_grad_()

            with torch.enable_grad():
                logits_adv = self.model(x_adv)
                loss_kl = criterion_kl(F.log_softmax(logits_adv, dim=1),
                                       F.softmax(logits_nat, dim=1))
                if i_epoch >= self.epoch_start_te:
                    loss = torch.sum((lams / batch_size).cuda() * torch.sum(loss_kl, dim=-1)) \
                           + reg_weight * ((F.softmax(logits_adv, dim=1)
                                            - soft_labels_batch) ** 2).mean()
                else:
                    loss = torch.sum(loss_kl)

            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        self.model.train()

        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        # zero gradient
        optimizer.zero_grad()

        logits_nat = self.model(x_natural)
        loss_natural = F.cross_entropy(logits_nat, y)

        logits_adv = self.model(x_adv)
        loss_robust = criterion_kl(F.log_softmax(logits_adv, dim=1),
                                   F.softmax(logits_nat, dim=1))
        loss_robust = lams.cuda() * torch.sum(loss_robust, dim=-1)
        loss_robust = torch.mean(loss_robust)

        loss = loss_natural + loss_robust

        if i_epoch >= self.epoch_start_te:
            loss = loss + \
                   reg_weight * ((F.softmax(logits_nat, dim=1)
                                  - soft_labels_batch) ** 2).mean()

        return loss, logits_adv.detach().cpu()
