#include "kernel/quant_matmul.hpp"

#include <acl/acl.h>
#include <acl/acl_rt.h>

#include "catlass_kernel.h"
#include "common.hpp"
#include "helper.hpp"

namespace CatlassKernel {
using namespace Catlass;

void QuantMatmul(uint32_t blockNum, aclrtStream stream, KernelInfo kernelInfo)
{
    // This kernel implements int8 input and bfloat16 output
    if (kernelInfo.inputDataType == ACL_INT8 && kernelInfo.outputDataType == ACL_BF16) {
        // The example uses RowMajor for A and ColumnMajor for B
        using LayoutA = layout::RowMajor;
        using LayoutB = layout::ColumnMajor;

        GemmCoord problemShape{kernelInfo.m, kernelInfo.n, kernelInfo.k};

        // Prepare FFTS address for synchronization
        uint64_t fftsAddr{0};
        uint32_t fftsLen{0};
        rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);

        // inputAddr should contain gmA, gmB, gmScale, gmPerTokenScale in order
        // outputAddr should contain gmD
        // workspaceAddr should contain gmWorkspace
        if (kernelInfo.inputAddr.size() < 4 || kernelInfo.outputAddr.empty() || kernelInfo.workspaceAddr == nullptr) {
            // Handle error: insufficient addresses provided
            return;
        }

        quant_matmul<LayoutA, LayoutB><<<blockNum, nullptr, stream>>>(
            fftsAddr,
            problemShape,
            kernelInfo.inputAddr.at(0),   // gmA
            kernelInfo.inputAddr.at(1),   // gmB
            kernelInfo.inputAddr.at(2),   // gmScale
            kernelInfo.inputAddr.at(3),   // gmPerTokenScale
            kernelInfo.outputAddr.at(0),  // gmD
            kernelInfo.workspaceAddr      // gmWorkspace
        );
    }
}
} // namespace CatlassKernel