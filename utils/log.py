import sys
import time
from pathlib import Path

from config import dir_wandb_saved_files
from utils.helper_funcs_wandb import get_wandb_dir_stem


class Logger(object):
    def __init__(self, dir_log: Path = Path('.'), name_log: str = "run_info.log", is_use_wandb=False):
        self.terminal = sys.stdout
        log_file_path = self.get_log_file_path(dir_log, name_log, is_use_wandb)
        self.log_file = open(log_file_path, "a")

    @staticmethod
    def get_log_file_path(dir_log: Path, name_log: str, is_use_wandb: bool):

        if is_use_wandb:
            run_dir_name = get_wandb_dir_stem()
            dir_log = dir_wandb_saved_files.joinpath(run_dir_name)

        if not dir_log.exists():
            dir_log.mkdir(exist_ok=True, parents=True)

        return dir_log.joinpath(name_log)

    def add_time(self, message: str):
        now_time = time.strftime("%Y-%m-%d %H:%M:%S")
        message = f"[{now_time}] {message}\n"
        return message

    def log(self, message: str):
        "log to file and terminal"
        message_with_time = self.add_time(message)
        self.terminal.write(message_with_time)
        self.log_file.write(message_with_time)
        self.flush()

    def log_to_file(self, message):
        message_with_time = self.add_time(message)
        self.log_file.write(message_with_time)
        self.flush()

    def flush(self):
        self.log_file.flush()

    def close(self):
        if not self.log_file.closed:
            self.log_file.close()
