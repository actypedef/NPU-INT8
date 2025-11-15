"""
Python bindings for CatlassKernel
"""
from __future__ import annotations
import torch
__all__: list[str] = ['basic_matmul', 'grouped_matmul', 'optimized_matmul', 'quant_matmul']
def basic_matmul(arg0: torch.Tensor, arg1: torch.Tensor, arg2: str) -> torch.Tensor:
    ...
def grouped_matmul(arg0: list[torch.Tensor], arg1: list[torch.Tensor], arg2: str, arg3: bool) -> list[torch.Tensor]:
    ...
def optimized_matmul(arg0: torch.Tensor, arg1: torch.Tensor, arg2: str) -> torch.Tensor:
    ...
def quant_matmul(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor) -> torch.Tensor:
    ...
