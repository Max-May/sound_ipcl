import yaml
from pathlib import Path
import numpy as np
import torch
import random


def read_yaml(fn: str):
    fn = Path(fn)
    with fn.open('r') as handle:
        return yaml.safe_load(handle)


def write_yaml(content, fn: str):
    fn = Path(fn)
    with fn.open('wt') as handle:
        yaml.dump(content, handle, indent=4)


def seed_all(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)