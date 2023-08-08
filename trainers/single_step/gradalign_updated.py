"""
Understanding and Improving Fast Adversarial Training
https://github.com/tml-epfl/understanding-fast-adv-training
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import nn
from tqdm import tqdm

from config import Config
from utils.helper_funcs_wandb import define_wandb_batch_metrics


class GradAlignUpdated:
    def __init__(self, config: Config, device: str, lr_scheduler,
                 model: torch.nn.Module, train_loader, optimizer, num_classes: int, targets,
                 use_wandb=False, **kwargs):
        self.model = model
        self.adv_config = config.param_atk_train
        self.optimizer = optimizer
        self.device = device
        self.total_epoch = config.total_epoch
        self.train_loader = train_loader
        self.grad_align_cos_lambda = 0.2

        self.record_cos_value = torch.zeros((len(targets),), requires_grad=False)

        # different for svhn dataset:
        # https://github.com/tml-epfl/understanding-fast-adv-training/blob/ceed49440bfc35a676375b783eaba5940e583bcc/train.py#L168
        # self.fgsm_alpha = 1.25

        self.epoch_start_collect = 3
        self.mometum_factor = 0.9
        self.num_cls = num_classes
        self.num_samples = len(targets)
        self.probs_global = torch.zeros((self.num_samples,), dtype=torch.float)
        self.logits_current_epoch = torch.zeros((self.num_samples, num_classes))
        self.targets = torch.from_numpy(targets)
        self.step_size_min = config.param_step_size_range.step_size_min
        self.step_size_max = config.param_step_size_range.step_size_max

        self.lr_scheduler = lr_scheduler

        self.bs_print = 200
        self.tqdm_bar = tqdm(total=len(self.train_loader) * self.total_epoch, file=sys.stdout, position=0, ncols=120)

        self.use_wandb = use_wandb
        if self.use_wandb:  # use wandb to record
            self.wb_metric_batch, self.wb_metric_lr, self.wb_metric_epoch, self.wb_metric_loss = define_wandb_batch_metrics()

    def _assign_step_size_to_samples(self):
        ids_sorted = torch.argsort(self.probs_global)  # probs rise
        step_sizes_for_samples = torch.empty((self.num_samples,), dtype=torch.float)
        step_sizes = torch.linspace(self.step_size_min, self.step_size_max, steps=self.num_samples)
        step_sizes_for_samples[ids_sorted] = step_sizes

        return step_sizes_for_samples

    def _collect_global_probs(self):
        probs_all = F.softmax(self.logits_current_epoch, dim=-1)
        probs = probs_all[range(self.num_samples), self.targets]
        self.probs_global = self.mometum_factor * self.probs_global + (1 - self.mometum_factor) * probs

    def train_epoch(self, idx_epoch):
        " idx_epoch should start from 1"
        if idx_epoch < self.epoch_start_collect:
            step_sizes_for_samples = torch.full((self.num_samples,), fill_value=self.step_size_min)
        else:
            self._collect_global_probs()
            step_sizes_for_samples = self._assign_step_size_to_samples()

        self.model.train()

        for batch_idx, (idx, data, target) in enumerate(self.train_loader, 1):
            data, target = data.to(self.device), target.to(self.device)

            lr = self.lr_scheduler.get_last_lr()[0]

            self.optimizer.zero_grad()
            step_sizes_batch = step_sizes_for_samples[idx]

            # calculate robust loss
            loss, logits_adv = self._train_batch(x_natural=data, y=target, step_sizes_batch=step_sizes_batch,
                                                 idx_samples=idx, optimizer=self.optimizer)

            if self.epoch_start_collect - 1 <= idx_epoch:
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
                                          f'lr-{lr:.3f} ')

            if self.use_wandb:
                idx_batch_total = (idx_epoch - 1) * len(self.train_loader) + batch_idx
                wandb.log({self.wb_metric_lr: lr, self.wb_metric_batch: idx_batch_total})
                wandb.log({self.wb_metric_loss: loss.item(), self.wb_metric_batch: idx_batch_total})
                wandb.log({self.wb_metric_epoch: idx_epoch, self.wb_metric_batch: idx_batch_total})

        if idx_epoch >= self.total_epoch:
            self.tqdm_bar.clear()
            self.tqdm_bar.close()

    @staticmethod
    def get_uniform_delta(shape, eps, requires_grad=True):
        delta = torch.zeros(shape).cuda()
        delta.uniform_(-eps, eps)
        delta.requires_grad = requires_grad
        return delta

    @staticmethod
    def clamp(X, l, u, cuda=True):
        if type(l) is not torch.Tensor:
            if cuda:
                l = torch.cuda.FloatTensor(1).fill_(l)
            else:
                l = torch.FloatTensor(1).fill_(l)
        if type(u) is not torch.Tensor:
            if cuda:
                u = torch.cuda.FloatTensor(1).fill_(u)
            else:
                u = torch.FloatTensor(1).fill_(u)
        return torch.max(torch.min(X, u), l)

    @staticmethod
    def l2_norm_batch(v):
        norms = (v ** 2).sum([1, 2, 3]) ** 0.5
        return norms

    def get_input_grad(self, X, y, delta_init='none', backprop=False):
        if delta_init == 'none':
            delta = torch.zeros_like(X, requires_grad=True)
        elif delta_init == 'random_uniform':
            delta = self.get_uniform_delta(X.shape, self.adv_config.epsilon, requires_grad=True)
        elif delta_init == 'random_corner':
            delta = self.get_uniform_delta(X.shape, self.adv_config.epsilon, requires_grad=True)
            delta = self.adv_config.epsilon * torch.sign(delta)
        else:
            raise ValueError('wrong delta init')

        output = self.model(X + delta)
        loss = F.cross_entropy(output, y)

        grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]
        if not backprop:
            grad, delta = grad.detach(), delta.detach()
        return grad

    def _train_batch(self, x_natural, y, step_sizes_batch, idx_samples, optimizer):
        criterion = nn.CrossEntropyLoss()

        batch_size = len(x_natural)
        step_sizes_batch = step_sizes_batch.view(batch_size, 1, 1, 1).cuda()

        delta = self.get_uniform_delta(x_natural.shape, self.adv_config.epsilon, requires_grad=True)
        x_adv = self.clamp(x_natural + delta, 0, 1)
        logits_adv = self.model(x_adv)
        loss_adv = criterion(logits_adv, y)
        grad = torch.autograd.grad(loss_adv, delta, create_graph=True)[0]
        grad = grad.detach()

        delta.data = self.clamp(delta.data + step_sizes_batch * torch.sign(grad), -self.adv_config.epsilon,
                                self.adv_config.epsilon)
        delta.data = self.clamp(x_natural + delta.data, 0, 1) - x_natural

        delta = delta.detach()

        logits_adv = self.model(x_natural + delta)
        loss_adv = criterion(logits_adv, y)

        reg = torch.zeros(1).cuda()[0]  # for .item() to run correctly

        grad2 = self.get_input_grad(x_natural, y, delta_init='random_uniform', backprop=True)
        grads_nnz_idx = ((grad ** 2).sum([1, 2, 3]) ** 0.5 != 0) * ((grad2 ** 2).sum([1, 2, 3]) ** 0.5 != 0)
        grad1, grad2 = grad[grads_nnz_idx], grad2[grads_nnz_idx]
        grad1_norms, grad2_norms = self.l2_norm_batch(grad1), self.l2_norm_batch(grad2)
        grad1_normalized = grad1 / grad1_norms[:, None, None, None]
        grad2_normalized = grad2 / grad2_norms[:, None, None, None]
        cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))
        reg += self.grad_align_cos_lambda * (1.0 - cos.mean())

        loss_adv += reg

        optimizer.zero_grad()  # TODO: necessary?

        # for record
        self.record_cos_value[idx_samples] += cos.detach().cpu()

        return loss_adv, logits_adv.detach().cpu()
