import torch
import torch_npu
import torch_catlass

def verify_quant_matmul_intuitive():
    """
    一个用于直观验证 quant_matmul 的小型示例。
    它会打印出所有输入、标准答案的详细计算步骤以及 Catlass Kernel 的最终输出。
    """
    print("=" * 60)
    print("--- Verifying torch_catlass.quant_matmul with a small example ---")
    print("=" * 60)

    # 1. 定义一个非常小的矩阵维度
    m, k, n = 2, 2, 2

    # 2. 准备输入张量，使用简单且易于手算的整数
    # mat_a: (m, k), int8, 行主序
    mat_a_data = [[1, 2], 
                  [3, 4]]
    mat_a = torch.tensor(mat_a_data, dtype=torch.int8).npu()

    # mat_b: (k, n), int8, 列主序.
    # 通过创建一个 (n, k) 的行主序张量然后转置 (.T) 得到
    mat_b_data_pre_transpose = [[10, -1],  # 逻辑上的 mat_b 的第一行
                                [0,   1]]   # 逻辑上的 mat_b 的第二行
    mat_b = torch.tensor(mat_b_data_pre_transpose, dtype=torch.int8).npu().T

    # scale 和 per_token_scale: 使用简单的浮点数
    scale_data = [0.5, 1.0]
    scale = torch.tensor(scale_data, dtype=torch.bfloat16).npu()

    per_token_scale_data = [1.0, 0.25]
    per_token_scale = torch.tensor(per_token_scale_data, dtype=torch.bfloat16).npu()

    # --- 打印所有输入，以便核对 ---
    print("\n--- Inputs ---")
    print(f"mat_a (int8, shape={mat_a.shape}):\n{mat_a}\n")
    # 为了直观，我们打印 mat_b 转置前的样子，这才是它逻辑上的样子
    print(f"mat_b (int8, logical shape=({k},{n})):\n{mat_b.T.cpu()}\n")
    print(f"scale (bf16, shape={scale.shape}):\n{scale}\n")
    print(f"per_token_scale (bf16, shape={per_token_scale.shape}):\n{per_token_scale}\n")

    # --- 详细计算标准答案 (Golden Result) ---
    print("\n--- Golden Calculation (Step-by-Step in float32 for precision) ---")
    
    # 为了保证中间计算的精度，我们先将所有输入转为 float32
    mat_a_fp32 = mat_a.to(torch.float32)
    mat_b_fp32 = mat_b.to(torch.float32)
    scale_fp32 = scale.to(torch.float32)
    per_token_scale_fp32 = per_token_scale.to(torch.float32)
    
    # 步骤 1: 整数矩阵乘法
    # [[1, 2],   @ [[10,  0],  = [[1*10+2*(-1), 1*0+2*1],   = [[8,  2],
    #  [3, 4]]      [-1,  1]]     [3*10+4*(-1), 3*0+4*1]]      [26, 4]]
    intermediate_mm = torch.mm(mat_a_fp32, mat_b_fp32)
    print(f"Step 1: mat_a @ mat_b:\n{intermediate_mm}\n")

    # 步骤 2: 应用 per_token_scale (按行缩放)
    # [[8,  2],  * [[1.0],   = [[8*1.0,  2*1.0],   = [[8.0,  2.0],
    #  [26, 4]]     [0.25]]    [26*0.25, 4*0.25]]     [6.5,  1.0]]
    intermediate_dequant1 = intermediate_mm * per_token_scale_fp32.unsqueeze(1)
    print(f"Step 2: Apply per_token_scale (row-wise):\n{intermediate_dequant1}\n")

    # 步骤 3: 应用 scale (按列缩放)
    # [[8.0, 2.0], * [[0.5, 1.0]] = [[8.0*0.5, 2.0*1.0], = [[4.0,  2.0],
    #  [6.5, 1.0]]                  [6.5*0.5, 1.0*1.0]]    [3.25, 1.0]]
    golden_fp32 = intermediate_dequant1 * scale_fp32.unsqueeze(0)
    print(f"Step 3: Apply scale (column-wise), Final Golden in fp32:\n{golden_fp32}\n")

    # 最终的标准答案需要转换回 bfloat16 类型
    golden = golden_fp32.to(torch.bfloat16)
    
    # --- 调用 Catlass Kernel 并获取结果 ---
    print("\n--- Calling Catlass Kernel ---")
    catlass_result = torch_catlass.quant_matmul(mat_a, mat_b, scale, per_token_scale)
    
    # --- 最终结果对比 ---
    print("\n" + "=" * 60)
    print("--- Final Comparison ---")
    print("=" * 60)
    print(f"Golden Result (bf16):\n{golden.cpu()}\n")
    print(f"Catlass Kernel Result (bf16):\n{catlass_result.cpu()}\n")

    # 使用 torch.allclose 进行程序化验证
    is_close = torch.allclose(golden, catlass_result)
    if is_close:
        print("SUCCESS: The results are numerically close.")
    else:
        print("FAILED: The results are different.")
    print("=" * 60)


if __name__ == "__main__":
    # 确保 NPU 设备可用
    if torch.npu.is_available():
        torch.npu.set_device(0)
        verify_quant_matmul_intuitive()
    else:
        print("NPU device not found. This script requires a Huawei NPU.")