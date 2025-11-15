/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef SHARED_LIB_IMPL_QUANT_MATMUL_H
#define SHARED_LIB_IMPL_QUANT_MATMUL_H

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/quant_matmul_multistage_workspace.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

namespace Catlass {

template <
    class LayoutA,
    class LayoutB
>
CATLASS_GLOBAL
void quant_matmul(
    uint64_t fftsAddr,
    GemmCoord problemShape,
    GM_ADDR gmA,
    GM_ADDR gmB,
    GM_ADDR gmScale,
    GM_ADDR gmPerTokenScale,
    GM_ADDR gmD,
    GM_ADDR gmWorkspace
)
{
    AscendC::SetSyncBaseAddr(fftsAddr);
    using ArchTag = Arch::AtlasA2;
    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;
    constexpr bool enableUnitFlag = false;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsyncWithCallback<
        preloadStages,
        l1Stages, l0AStages, l0BStages, l0CStages,
        enableUnitFlag, enableShuffleK
    >;
    using L1TileShape = GemmShape<128, 256, 512>;
    using L0TileShape = GemmShape<128, 256, 128>;
    constexpr uint32_t workspaceStages = 2;

    using AType = Gemm::GemmType<int8_t, LayoutA>;
    using BType = Gemm::GemmType<int8_t, LayoutB>;
    using CType = Gemm::GemmType<int32_t, layout::RowMajor>;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    constexpr uint32_t ubStages = 2;
    using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
    using ScaleType = Gemm::GemmType<bfloat16_t, layout::VectorLayout>;
    using PerTokenScaleType = Gemm::GemmType<bfloat16_t, layout::VectorLayout>;
    using DType = Gemm::GemmType<bfloat16_t, layout::RowMajor>;

    using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;

    using EpilogueTileShape = MatrixShape<32, 256>;
    using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
    using TileBroadcastOneBlk = Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType,
        EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul = Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag,
        OneBlkColumnBroadcastMulType, EpilogueTileShape>;
    using TileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, DType>;
    using BlockScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using BlockEpilogue = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType, ScaleType, PerTokenScaleType,
        DType, TileRowBroadcastMul, TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul, TileCopy, BlockScheduler>;

    LayoutA layoutA(problemShape.m(), problemShape.k());
    LayoutB layoutB(problemShape.k(), problemShape.n());
    layout::VectorLayout layoutScale(problemShape.n());
    layout::VectorLayout layoutPerTokenScale(problemShape.m());
    layout::RowMajor layoutD(problemShape.m(), problemShape.n());

    if (problemShape.m() > problemShape.n()) {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        using MatmulKernel = Gemm::Kernel::QuantMatmulMultiStageWorkspace<BlockMmad, BlockEpilogue,
            BlockScheduler, workspaceStages>;
        typename MatmulKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmScale, layoutScale,
            gmPerTokenScale, layoutPerTokenScale, gmD, layoutD, gmWorkspace};
        MatmulKernel matmul;
        matmul(params);
    } else {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
        using MatmulKernel = Gemm::Kernel::QuantMatmulMultiStageWorkspace<BlockMmad, BlockEpilogue,
            BlockScheduler, workspaceStages>;
        typename MatmulKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmScale, layoutScale,
            gmPerTokenScale, layoutPerTokenScale, gmD, layoutD, gmWorkspace};
        MatmulKernel matmul;
        matmul(params);
    }
}

} // namespace Catlass
#endif // SHARED_LIB_IMPL_QUANT_MATMUL_H