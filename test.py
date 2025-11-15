import torch_catlass
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

class CatlassTest(TestCase):
    def test_basic_matmul(self):
        a = torch.ones((2, 3)).to(torch.float16).npu()
        b = torch.ones((3, 4)).to(torch.float16).npu()
        result = torch_catlass.basic_matmul(a, b, "float16")
        golden = torch.mm(a, b)
        self.assertRtolEqual(result, golden)

    def test_quant_matmul(self):
        """
        测试 INT8 GEMM (Quant Matmul) 功能。
        该算子要求 mat_a(int8) @ mat_b(int8) -> output(bf16)，
        并使用 per_token_scale 和 scale 两个 bf16 类型的张量进行反量化。
        """
        m, k, n = 128, 4096, 4096

        # mat_a: (m, k), int8, 行主序 (contiguous)
        mat_a = torch.randint(-16, 16, (m, k), dtype=torch.int8).npu()

        # mat_b: (k, n), int8, 列主序.
        mat_b = torch.randint(-16, 16, (n, k), dtype=torch.int8).npu().T
        
        # scale: (n,), bfloat16
        scale = torch.rand((n,), dtype=torch.bfloat16).npu()
        
        # per_token_scale: (m,), bfloat16
        per_token_scale = torch.rand((m,), dtype=torch.bfloat16).npu()

        result = torch_catlass.quant_matmul(mat_a, mat_b, scale, per_token_scale)

        golden_fp32 = torch.mm(mat_a.to(torch.float32), mat_b.to(torch.float32))
        
        # 应用 per-token 和 per-channel scales.
        golden_fp32 = golden_fp32 * per_token_scale.to(torch.float32).unsqueeze(1)
        golden_fp32 = golden_fp32 * scale.to(torch.float32).unsqueeze(0)

        # 将最终结果转换为 bfloat16
        golden = golden_fp32.to(torch.bfloat16)

        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertRtolEqual(result, golden)
        
if __name__ == "__main__":
    run_tests()