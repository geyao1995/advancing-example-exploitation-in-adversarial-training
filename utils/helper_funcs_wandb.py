import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import get_type_hints

import numpy as np
import torch
import wandb

from config import Config, dir_wandb, dir_wandb_saved_files


def set_wandb_env(is_online: bool = True):
    # https://docs.wandb.ai/guides/track/advanced/environment-variables

    # ignore files when upload, no spaces after the comma separator. no effect in syncing offline run
    os.environ['WANDB_IGNORE_GLOBS'] = '*.pth,*.npy'

    # https://github.com/wandb/wandb/issues/4872
    os.environ['WANDB_DISABLE_SERVICE'] = 'true'

    # https://github.com/wandb/wandb/issues/4872#issuecomment-1459094979
    os.environ['WANDB_CONSOLE'] = 'off'

    if is_online:
        os.environ['WANDB_MODE'] = 'online'
    else:
        # can be later synced with the `wandb sync` command.
        os.environ['WANDB_MODE'] = 'offline'


def disable_debug_internal_log():
    # https://community.wandb.ai/t/the-debug-internal-log-file-is-too-large-500mb/3589/1
    # https://github.com/wandb/wandb/issues/4223

    if not dir_wandb.exists():
        dir_wandb.mkdir(parents=True)

    if dir_wandb.joinpath('null').exists():
        return
    else:
        os.system(f"ln -s /dev/null {str(dir_wandb)}/null")


def define_wandb_epoch_metrics():
    epoch_step = 'epoch_step'
    test_acc = 'test/acc'
    test_rob = 'test/rob'
    wandb.define_metric(epoch_step)
    wandb.define_metric(test_acc, step_metric=epoch_step, summary='max')
    wandb.define_metric(test_rob, step_metric=epoch_step, summary='max')

    return epoch_step, test_acc, test_rob


def define_wandb_batch_metrics():
    batch_step = 'batch_step'
    train_lr = 'train/lr'
    loss = 'train/loss'
    epoch = 'train/epoch'

    wandb.define_metric(batch_step)
    wandb.define_metric(train_lr, step_metric=batch_step)
    wandb.define_metric(loss, step_metric=batch_step, summary='min')
    wandb.define_metric(epoch, step_metric=batch_step)

    return batch_step, train_lr, epoch, loss


def make_wandb_config(my_config: Config):
    wandb_config = asdict(my_config)

    return wandb_config


def from_wandb_config(cfg: dict) -> Config:
    """convert wandb config to dataclass config"""
    config_new = Config(**cfg)
    hints = get_type_hints(Config)
    for name_var, type_var in hints.items():

        if is_dataclass(type_var):
            if cfg[name_var] is not None:
                config_new.__dict__[name_var] = type_var(**cfg[name_var])

    return config_new


def get_wandb_dir_stem() -> str:
    return Path(wandb.run.dir).parent.stem


def save_model_to_wandb_dir(model: torch.nn.Module, idx_epoch: int):
    run_dir_name = get_wandb_dir_stem()
    weights_path = dir_wandb_saved_files.joinpath(run_dir_name, 'weight')

    weights_path.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), weights_path.joinpath(f'epoch_{idx_epoch}.pth'))
    print(f'Save epoch-{idx_epoch} checkpoints to {weights_path}')


def save_array_to_wandb_dir(array: np.ndarray, name: str):
    run_dir_name = get_wandb_dir_stem()
    array_path = dir_wandb_saved_files.joinpath(run_dir_name, 'array')
    array_path.mkdir(exist_ok=True, parents=True)
    np.save(array_path.joinpath(name).__str__(), array)
    print(f'Save array {name} to {array_path}')


def get_run_obj(run_instance):
    """
    :param run_instance: string, <entity>/<project>/<run_id>
    :return: run object
    """
    # https://docs.wandb.ai/ref/python/public-api/api
    api = wandb.Api()
    run = api.run(run_instance)

    return run


def export_wandb_run_data(run_instance, keys):
    """
    :param run_instance: string, <entity>/<project>/<run_id>
    :param is_from_cloud: bool
    :return:
    """
    run = get_run_obj(run_instance)

    if run.state == "finished":
        # https://github.com/wandb/wandb/blob/latest/wandb/apis/public.py#L1968
        return run.history(keys=keys, pandas=False)
    else:
        print('Run is not finished!')


def export_wandb_run_config(run_instance):
    run = get_run_obj(run_instance)

    if run.state == "finished":
        return run.config
    else:
        print('Run is not finished!')


def export_files_name(run_instance):
    run = get_run_obj(run_instance)

    if run.state == "finished":
        return [i for i in run.files()]
        # download
        # for file in run.files():
        #     file.download()
    else:
        print('Run is not finished!')


def update_wandb_exist_run_config(run_path: str, key, value):
    """
    run_path: like geyao/Salient Feature Influence Test/2m4mcw7b
    """
    api = wandb.Api()
    run = api.run(run_path)
    run.config[key] = value
    run.update()


def update_wandb_group_name(name_project: str, name_old_group: str, name_new_group: str):
    api = wandb.Api()
    runs_in_project = api.runs(name_project)
    for run in runs_in_project:
        # print(run.name)
        if run.group == name_old_group:
            run.group = name_new_group
        run.update()


def get_wandb_run_path(run_id: str):
    " get local dir according to wandb run id "
    run_dir_mathc_list = [str(i) for i in dir_wandb_saved_files.glob(f'*-{run_id}')]
    assert len(run_dir_mathc_list) == 1, 'Should be single match result, check it!'

    return next(dir_wandb_saved_files.glob(f'*-{run_id}'))


def get_path_for_checkpoints(run_id: str):
    run_dir = get_wandb_run_path(run_id)
    checkpoints_path_list = [i for i in run_dir.joinpath('files', 'weights').glob('*.pth')]
    checkpoints_path_list = sorted(checkpoints_path_list, key=lambda x: int(x.stem.split('_')[-1]))
    return checkpoints_path_list


def check_and_get_path_for_single_checkpoint(run_id: str, num_epoch: int):
    run_dir = get_wandb_run_path(run_id)
    checkpoint_path = run_dir.joinpath('weight', f'epoch_{num_epoch}.pth')
    if checkpoint_path.exists():
        is_checkpoint_exist = True
    else:
        is_checkpoint_exist = False
    return is_checkpoint_exist, checkpoint_path
