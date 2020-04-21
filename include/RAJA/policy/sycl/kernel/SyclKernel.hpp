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

/*!
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
template <bool async0, size_t num_blocks, size_t num_threads>
struct sycl_launch {};

/*!
 * SYCL kernel launch policy where the user specifies the number of physical
 * thread blocks and threads per block.
 * If num_blocks is 0 then num_blocks is chosen at runtime.
 * Num_blocks is chosen to maximize the number of blocks running concurrently.
 */
template <bool async0, size_t num_blocks, size_t num_threads>
using sycl_explicit_launch = sycl_launch<async0, num_blocks, num_threads>;

namespace statement
{

/*!
 * A RAJA::kernel statement that launches a CUDA kernel.
 *
 *
 */
template <typename LaunchConfig, typename... EnclosedStmts>
struct SyclKernelExt
    : public internal::Statement<sycl_exec<0>, EnclosedStmts...> {
};

/*!
 * A RAJA::kernel statement that launches a SYCL kernel with a fixed
 * number of threads (specified by num_threads)
 * The kernel launch is asynchronous.
 */
template <size_t num_threads, typename... EnclosedStmts>
using SyclKernelFixedAsync =
    SyclKernelExt<sycl_explicit_launch<true, operators::limits<size_t>::max(), num_threads>,
                  EnclosedStmts...>;


/*!
 *  * A RAJA::kernel statement that launches a CUDA kernel with 1024 threads
 *   * The kernel launch is synchronous.
 *    */
template <typename... EnclosedStmts>
using SyclKernel = SyclKernelFixedAsync<1024, EnclosedStmts...>;
}  // namespace statement

namespace internal
{

/*!
 * SYCL global function for launching SyclKernel policies
 * This is annotated to guarantee that device code generated
 * can be launched by a kernel with BlockSize number of threads.
 *
 * This launcher is used by the SyclKerelFixed policies.
 */
template <size_t BlockSize, typename Data, typename Exec>
//__launch_bounds__(BlockSize, 1) __global__
    void SyclKernelLauncherFixed(Data data, cl::sycl::nd_item<3> item)
{

  using data_t = camp::decay<Data>;
  data_t private_data = data;

  // execute the the object
  Exec::exec(private_data, item);
}

/*!
 * Helper class that handles getting the correct global function for
 * SyclKernel policies. This class is specialized on whether or not BlockSize
 * is fixed at compile time.
 *
 * The default case handles BlockSize != 0 and gets the fixed max block size
 * version of the kernel.
 */
template<size_t BlockSize, typename Data, typename executor_t>
struct SyclKernelLauncherGetter
{
  using type = camp::decay<decltype(&internal::SyclKernelLauncherFixed<BlockSize, Data, executor_t>)>;
  static constexpr type get() noexcept
  {
    return internal::SyclKernelLauncherFixed<BlockSize, Data, executor_t>;
  }
};

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
template<bool async0, size_t num_blocks, size_t num_threads, typename StmtList, typename Data, typename Types>
struct SyclLaunchHelper<sycl_launch<async0, num_blocks, num_threads>,StmtList,Data,Types>
{
  using Self = SyclLaunchHelper;

  static constexpr bool async = async0;

  using executor_t = internal::sycl_statement_list_executor_t<StmtList, Data, Types>;

  using kernelGetter_t = SyclKernelLauncherGetter<(num_threads <= 0) ? 0 : num_threads, Data, executor_t>;

  using data_t = camp::decay<Data>;

  static void launch(Data &&data,
                     internal::LaunchDims launch_dims,
                     size_t shmem,
                     cl::sycl::queue stream)
  {
    auto func = kernelGetter_t::get();

    void *args[] = {(void*)&data};

    data_t* d_data = (data_t*) malloc_device(sizeof(data_t), stream);
    auto e = stream.memcpy(d_data, &data, sizeof(data_t));
    e.wait();

    std::cout << "Blocks.x = " << launch_dims.blocks.x
              << "\nBlocks.y = " << launch_dims.blocks.y
              << "\nBlocks.z = " << launch_dims.blocks.z
              << "\nThreads.x = " << launch_dims.threads.x
              << "\nThreads.y = " << launch_dims.threads.y
              << "\nThreads.z = " << launch_dims.threads.z;


    cl::sycl::nd_range<3> range = {{launch_dims.blocks.x, launch_dims.blocks.y, launch_dims.blocks.z},
                                  {launch_dims.threads.x, launch_dims.threads.y, launch_dims.threads.z}};

    stream.submit([&](cl::sycl::handler& h) {
 
      h.parallel_for( range,
                      [=] (cl::sycl::nd_item<3> item) {

//      size_t ii = item.get_global_id(0);
       SyclKernelLauncherFixed<256, Data, executor_t>(*d_data,item);
      });
    });

    stream.wait();
//    if (!async) { stream.wait(); };

    cl::sycl::free(d_data, stream);
    

//    RAJA::sycl::launch(func, launch_dims.blocks, launch_dims.threads, args, shmem, stream);
  }
};
// SyclLaunchHelper, actually launches the kernel
// Also StatementExecutor with LaunchConfig


/*!
 *  * Specialization that launches SYCL kernels for RAJA::kernel from host code
 *   */
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

    std::cout << "entry into launch" << std::endl;

//    SyclForWrapper<Data, Types, EnclosedStmts...> for_wrapper(data);

    //
    // Compute the requested kernel dimensions
    //
    LaunchDims launch_dims = executor_t::calculateDimensions(data);
    
    //LaunchDims launch_dims;
 //   launch_dims.threads = segment_length<ArgumentId>(data); 
   // launch_dims.blocks = 256 * ((launch_dims.threads + 256 -1) / 256);

    int shmem = 0;
    cl::sycl::queue stream;


//    auto sycl_data = RAJA::sycl::make_launch_body(
//         launch_dims.blocks, launch_dims.threads, shmem, stream, data);
    //
    // Launch the kernels
    //
    launch_t::launch(std::move(data), launch_dims, shmem, stream);

  }

};

}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_SYCL guard

#endif  // closing endif for header file include guard
