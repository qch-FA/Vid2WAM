"""Small utility subset used by Evaluation."""

import os
import random
from typing import Callable, Dict, Optional

import numpy as np
import torch


def _resolve_global_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def set_global_seed(
    seed: int,
    get_worker_init_fn: bool = False,
) -> Optional[Callable[[int], None]]:
    if not np.iinfo(np.uint32).min < seed < np.iinfo(np.uint32).max:
        raise ValueError("Seed is outside uint32 bounds.")
    process_seed = int(seed) + _resolve_global_rank()
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
    if get_worker_init_fn:
        raise ValueError("Evaluation does not create DataLoader workers.")
    return None


def dict_apply(values: Dict, function: Callable) -> Dict:
    result = {}
    for key, value in values.items():
        result[key] = dict_apply(value, function) if isinstance(value, dict) else function(value)
    return result
