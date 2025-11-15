# NPU-INT8

NPU-INT8 provides python extension for INT8 GEMM kernels, then you can perform INT8 per-token quantization on Ascend NPUs. 


We have tested this on the 910B2 NPU, with CANN 8.3RC1. Make sure you run `set_env.sh` after Toolkit installation. See [CATLASS](https://gitcode.com/cann/catlass) for details. 


## Installation

```
pip install -r requirements.txt
```

```
cd kernels/
bash scripts/build.sh python_extension
```

```
pip install output/python_extension/torch_catlass-***.whl
```

## Usage

```
python test.py
```