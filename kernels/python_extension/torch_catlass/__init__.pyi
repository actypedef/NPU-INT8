from __future__ import annotations
import os as os
import sysconfig as sysconfig
import torch as torch
from torch_catlass._C import basic_matmul
from torch_catlass._C import grouped_matmul
from torch_catlass._C import optimized_matmul
from torch_catlass._C import quant_matmul
import torch_npu as torch_npu
from . import _C
__all__: list = list()
def _load_depend_libs():
    ...
