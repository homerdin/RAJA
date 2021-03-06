//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-kernel-region-sync.hpp"

#include "../../forall/test-forall-utils.hpp"

using OpenMPKernelRegionSyncExecPols = 
  camp::list< 

    RAJA::KernelPolicy<
      RAJA::statement::Region<RAJA::omp_parallel_region,

        RAJA::statement::For<0, RAJA::omp_for_nowait_exec,
          RAJA::statement::Lambda<0, RAJA::Segs<0>>
        >,

        RAJA::statement::OmpSyncThreads,

        RAJA::statement::For<1, RAJA::omp_for_nowait_exec,
          RAJA::statement::Lambda<1, RAJA::Segs<1>>
        >,

        RAJA::statement::OmpSyncThreads,

        RAJA::statement::For<0, RAJA::omp_for_nowait_exec,
          RAJA::statement::Lambda<2, RAJA::Segs<0>>
        >

      >
    >,

    RAJA::KernelPolicy<
      RAJA::statement::Region<RAJA::omp_parallel_region,

        RAJA::statement::For<0, RAJA::omp_for_exec,
          RAJA::statement::Lambda<0, RAJA::Segs<0>>
        >,

        RAJA::statement::OmpSyncThreads,

        RAJA::statement::For<1, RAJA::omp_for_exec,
          RAJA::statement::Lambda<1, RAJA::Segs<1>>
        >,

        RAJA::statement::OmpSyncThreads,

        RAJA::statement::For<0, RAJA::omp_for_exec,
          RAJA::statement::Lambda<2, RAJA::Segs<0>>
        >

      >
    >

  >;


// Cartesian product of types for OpenMP tests
using OpenMPKernelRegionSyncTypes =
  Test< camp::cartesian_product<IdxTypeList, 
                                HostResourceList,
                                OpenMPKernelRegionSyncExecPols> >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(OpenMP,
                               KernelRegionFunctionalTest,
                               OpenMPKernelRegionSyncTypes);
