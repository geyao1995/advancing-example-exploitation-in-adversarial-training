import sys

import torch.nn
import torch.nn.functional as F
import torch.optim as optim
from advertorch.attacks import LinfPGDAttack, CarliniWagnerL2Attack
from autoattack import AutoAttack
from torch.autograd import Variable
from tqdm import tqdm

from config import *
from utils.dataloaders import get_test_loader
from utils.helper_funcs import choose_model


class PgdMyAttack:
    def __init__(self, model, adv_config: ConfigLinfAttack):
        self.model = model
        self.epsilon = adv_config.epsilon
        self.perturb_steps = adv_config.perturb_steps
        self.step_size = adv_config.step_size

    def perturb(self, x, y, random_start=True):
        X_pgd = Variable(x.data, requires_grad=True)
        if random_start:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-self.epsilon, self.epsilon).cuda()
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for _ in range(self.perturb_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()

            with torch.enable_grad():
                loss = F.cross_entropy(self.model(X_pgd), y)
            loss.backward()
            eta = self.step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - x.data, -self.epsilon, self.epsilon)
            X_pgd = Variable(x.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

        return X_pgd.detach()


class PGD_50_10_Attack:
    def __init__(self, model, epsilon, step_size):
        """
        L_inf mode
        from: https://github.com/nblt/Sub-AT/blob/main/utils.py
        50 iterations with 10 random start
        """
        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.attack_iters = 50
        self.restarts = 10
        self.norm = 'l_inf'
        self.early_stop = False

    @staticmethod
    def _clamp(x, lower_limit, upper_limit):
        return torch.max(torch.min(x, upper_limit), lower_limit)

    def perturb(self, x, y):
        upper_limit, lower_limit = 1, 0
        max_loss = torch.zeros(y.shape[0]).cuda()
        max_delta = torch.zeros_like(x).cuda()
        for _ in range(self.restarts):
            delta = torch.zeros_like(x).cuda()
            if self.norm == "l_inf":
                delta.uniform_(-self.epsilon, self.epsilon)
            # elif norm == "l_2":
            #     delta.uniform_(-0.5, 0.5).renorm(p=2, dim=1, maxnorm=epsilon)
            else:
                raise ValueError
            delta = self._clamp(delta, lower_limit - x, upper_limit - x)
            delta.requires_grad = True
            for _ in range(self.attack_iters):
                output = self.model(x + delta)
                if self.early_stop:
                    index = torch.where(output.max(1)[1] == y)[0]
                else:
                    index = slice(None, None, None)
                if not isinstance(index, slice) and len(index) == 0:
                    break
                loss = F.cross_entropy(output, y)
                loss.backward()
                grad = delta.grad.detach()
                d = delta[index, :, :, :]
                g = grad[index, :, :, :]
                x = x[index, :, :, :]
                if self.norm == "l_inf":
                    d = torch.clamp(d + self.step_size * torch.sign(g), min=-self.epsilon,
                                    max=self.epsilon)
                # elif norm == "l_2":
                #     g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                #     scaled_g = g / (g_norm + 1e-10)
                #     d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
                d = self._clamp(d, lower_limit - x, upper_limit - x)
                delta.data[index, :, :, :] = d
                delta.grad.zero_()

            all_loss = F.cross_entropy(self.model(x + delta), y, reduction='none')
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)

        return x + max_delta


class RobEvaluator:
    def __init__(self, model: ModelName, data: DataName):
        self.model = choose_model(data, model)
        self.test_loader, self.num_classes = get_test_loader(data)

    def load_weights(self, weights_path: str):
        self.model.load_state_dict(torch.load(weights_path))

    def _select_eval_attack(self, eval_attack: RobEvalAttack, attack_param: ConfigLinfAttack):
        if eval_attack is RobEvalAttack.PGD_my:
            adversary = PgdMyAttack(self.model, attack_param)
            make_adv_batch = adversary.perturb
        elif eval_attack is RobEvalAttack.PGD_common:
            adversary = LinfPGDAttack(
                self.model, loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
                eps=attack_param.epsilon, nb_iter=attack_param.perturb_steps,
                eps_iter=attack_param.step_size,
                rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
            make_adv_batch = adversary.perturb
        elif eval_attack is RobEvalAttack.PGD_50_10:
            adversary = PGD_50_10_Attack(
                self.model, epsilon=attack_param.epsilon, step_size=attack_param.step_size)
            make_adv_batch = adversary.perturb
        elif eval_attack is RobEvalAttack.CW:
            adversary = CarliniWagnerL2Attack(predict=self.model, num_classes=self.num_classes,
                                              clip_min=0., clip_max=1.)
            make_adv_batch = adversary.perturb
        elif eval_attack is RobEvalAttack.Auto:
            adversary = AutoAttack(self.model, norm='Linf', eps=attack_param.epsilon, version='standard',
                                   verbose=False)

            def make_adv_batch(x, y):
                return adversary.run_standard_evaluation(x, y, bs=128)
        else:
            raise ValueError

        return make_adv_batch

    def eval_rob(self, eval_attack: RobEvalAttack, attack_param: ConfigLinfAttack):
        self.model = self.model.cuda()
        self.model.eval()
        make_adv_batch = self._select_eval_attack(eval_attack, attack_param)

        iterator_tqdm = tqdm(self.test_loader, file=sys.stdout, position=0, ncols=120)

        total_err = 0
        count = 0
        for x, y in iterator_tqdm:
            x, y = x.cuda(), y.cuda()
            x_adv = make_adv_batch(x, y)
            yp = self.model(x_adv)

            count += len(y)
            total_err += (yp.max(dim=1)[1] != y).sum().item()

            iterator_tqdm.set_description_str(f'On {count} examples.'
                                              f' Rob:{1 - total_err / count:.2%}')

        iterator_tqdm.close()

        return 1 - total_err / count
