import sys

import torch.nn
from tqdm import tqdm

from config import *
from utils.dataloaders import get_test_loader
from utils.helper_funcs import choose_model


class NatEvaluator:
    def __init__(self, model: ModelName, data: DataName):
        self.device = 'cuda'
        self.model = choose_model(data, model)
        self.test_loader, _ = get_test_loader(data)

    def load_weights(self, weights_path: str):
        self.model.load_state_dict(torch.load(weights_path))

    def eval_nat(self):
        self.model = self.model.cuda()
        self.model.eval()
        correct = 0
        total = 0

        iterator_tqdm = tqdm(self.test_loader, file=sys.stdout, position=0, ncols=120)

        with torch.no_grad():
            for i, test_batch in enumerate(iterator_tqdm):
                inputs = test_batch[0].to(self.device)
                labels = test_batch[1].to(self.device)

                outputs = self.model(inputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                iterator_tqdm.set_description_str(f'Test on {total} examples. '
                                                  f'Natural acc-{correct / total:.2%}')
        iterator_tqdm.close()

        return correct / total
