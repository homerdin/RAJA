/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel
 *          traversals on GPU with SYCL.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_sycl_kernel_SyclKernel_HPP
#define RAJA_policy_sycl_kernel_SyclKernel_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <cassert>
#include <climits>

#include "camp/camp.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel.hpp"
#include "RAJA/pattern/kernel/For.hpp"
#include "RAJA/pattern/kernel/Lambda.hpp"

#include "RAJA/policy/sycl/MemUtils_SYCL.hpp"
#include "RAJA/policy/sycl/policy.hpp"

#include "RAJA/policy/sycl/kernel/internal.hpp"

namespace RAJA
{

/*! TODO
 * SYCL kernel launch policy where the user may specify the number of physical
 * thread blocks and threads per block.
 * If num_blocks is 0 and num_threads is non-zero then num_blocks is chosen at
 * runtime.
 * Num_blocks is chosen to maximize the number of blocks running concurrently.
 * If num_threads and num_blocks are both 0 then num_threads and num_blocks are
 * chosen at runtime.
 * Num_threads and num_blocks are determined by the SYCL occupancy calculator.
 * If num_threads is 0 and num_blocks is non-zero then num_threads is chosen at
 * runtime.
 * Num_threads is 1024, which may not be appropriate for all kernels.
 */
template <bool async0>
struct sycl_launch {};

namespace statement
{

/*!
 * A RAJA::kernel statement that launches a SYCL kernel.
 *
 *
 */
template <typename LaunchConfig, typename... EnclosedStmts>
struct SyclKernelExt
    : public internal::Statement<sycl_exec<0>, EnclosedStmts...> {
};

/*!
 * A RAJA::kernel statement that launches a SYCL kernel.
 * The kernel launch is synchronous.
 */
template <typename... EnclosedStmts>
using SyclKernel =
    SyclKernelExt<sycl_launch<false>,
                  EnclosedStmts...>;

/*!
 * A RAJA::kernel statement that launches a SYCL kernel.
 * The kernel launch is asynchronous.
 */
template <typename... EnclosedStmts>
using SyclKernelAsync =
    SyclKernelExt<sycl_launch<true>,
                  EnclosedStmts...>;

} // namespace statement

namespace internal
{

/*!
 * SYCL global function for launching SyclKernel policies
 * This is annotated to guarantee that device code generated
 * can be launched by a kernel with BlockSize number of threads.
 *
 * This launcher is used by the SyclKerelFixed policies.
 */
template <typename Data, typename Exec>
void SyclKernelLauncher(Data data, cl::sycl::nd_item<3> item)
{

  using data_t = camp::decay<Data>;
  data_t private_data = data;

  // execute the the object
  Exec::exec(private_data, item, true);
}

/*!
 * Helper class that handles SYCL kernel launching, and computing
 * maximum number of threads/blocks
 */
template<typename LaunchPolicy, typename StmtList, typename Data, typename Types>
struct SyclLaunchHelper;


/*!
 * Helper class specialization to determine the number of threads and blocks.
 * The user may specify the number of threads and blocks or let one or both be
 * determined at runtime using the SYCL occupancy calculator.
 */
template<bool async0, typename StmtList, typename Data, typename Types>
struct SyclLaunchHelper<sycl_launch<async0>,StmtList,Data,Types>
{
  using Self = SyclLaunchHelper;

  static constexpr bool async = async0;

  using executor_t = internal::sycl_statement_list_executor_t<StmtList, Data, Types>;
  using data_t = camp::decay<Data>;

  static void launch(Data &&data,
                     internal::LaunchDims launch_dims,
                     size_t shmem,
                     cl::sycl::queue stream)
  {

    cl::sycl::buffer<data_t> d_data (std::move(&data), 1);

    stream.submit([&](cl::sycl::handler& h) {

      auto data = d_data.template get_access<cl::sycl::access::mode::read>(h);
 
      h.parallel_for(launch_dims.fit_nd_range(),
                     [=] (cl::sycl::nd_item<3> item) {
        
        SyclKernelLauncher<Data, executor_t>(data[0], item);

      });
    });

    if (!async) { stream.wait(); };

  }
};

/*!
 * Specialization that launches SYCL kernels for RAJA::kernel from host code
 */
template <typename LaunchConfig, typename... EnclosedStmts, typename Types>
struct StatementExecutor<
    statement::SyclKernelExt<LaunchConfig, EnclosedStmts...>, Types> {

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using StatementType =
      statement::SyclKernelExt<LaunchConfig, EnclosedStmts...>;

  template <typename Data>
  static inline void exec(Data &&data)
  {

    using data_t = camp::decay<Data>;
    using executor_t = sycl_statement_list_executor_t<stmt_list_t, data_t, Types>;
    using launch_t = SyclLaunchHelper<LaunchConfig, stmt_list_t, data_t, Types>;

    //
    // Compute the requested kernel dimensions
    //
    LaunchDims launch_dims = executor_t::calculateDimensions(data);
    
    int shmem = 0;
    cl::sycl::queue q = ::RAJA::sycl::detail::getQueue();

    //
    // Launch the kernels
    //
    launch_t::launch(std::move(data), launch_dims, shmem, q);

  }

};

}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_SYCL guard

#endif  // closing endif for header file include guard
