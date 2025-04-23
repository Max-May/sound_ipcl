import yaml
from pathlib import Path
import numpy as np
import torch
import random
import re


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


def count_pattern_files(pattern: str) -> int:
    """
    Returns the number of files represented by a string pattern.
    
    - If input is like '00' → returns 1.
    - If input is like '{00..09}' → returns 10.
    """
    match = re.match(r'\{(\d+)\.\.(\d+)\}', pattern)
    if match:
        start, end = match.groups()
        return int(end) - int(start) + 1
    else:
        return 1
