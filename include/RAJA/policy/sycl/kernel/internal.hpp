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

#ifndef RAJA_policy_sycl_kernel_internal_HPP
#define RAJA_policy_sycl_kernel_internal_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <cassert>
#include <climits>

#include "camp/camp.hpp"

#include "RAJA/pattern/kernel.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/sycl/MemUtils_SYCL.hpp"
#include "RAJA/policy/sycl/policy.hpp"

namespace RAJA
{

/*!
 * Policy for For<>, executes loop iteration by distributing them over threads
 * and blocks, but limiting the number of threads to num_threads.
 */
/*template <size_t num_threads>
struct sycl_threadblock_exec
    : public make_policy_pattern_launch_platform_t<
          Policy::sycl,
          Pattern::forall,
          Launch::undefined,
          Platform::sycl> {
};*/

namespace internal
{

// LaunchDims and Helper functions
struct LaunchDims {
  sycl_dim_3_t blocks;
  sycl_dim_3_t threads;
  sycl_dim_3_t global;

  RAJA_INLINE
  RAJA_HOST_DEVICE
  LaunchDims() : blocks{1,1,1}, // min_blocks{0},
                 threads{1,1,1},
                 global{1,1,1} {}//, min_threads{0} {}

  RAJA_INLINE
  RAJA_HOST_DEVICE
  LaunchDims(LaunchDims const &c) :
  blocks(c.blocks),//   min_blocks(c.min_blocks),
  threads(c.threads),//, min_threads(c.min_threads)
  global(c.global)
  {
  }

  RAJA_INLINE
  LaunchDims max(LaunchDims const &c) const
  {
    LaunchDims result;

    result.blocks.x = std::max(c.blocks.x, blocks.x);
    result.blocks.y = std::max(c.blocks.y, blocks.y);
    result.blocks.z = std::max(c.blocks.z, blocks.z);

    result.threads.x = std::max(c.threads.x, threads.x);
    result.threads.y = std::max(c.threads.y, threads.y);
    result.threads.z = std::max(c.threads.z, threads.z);

    result.global.x = std::max(c.global.x, global.x);
    result.global.y = std::max(c.global.y, global.y);
    result.global.z = std::max(c.global.z, global.z);

    return result;
  }

  cl::sycl::nd_range<3> fit_nd_range() {

    cl::sycl::queue q = ::RAJA::sycl::detail::getQueue();
    auto sizes = q.get_device().get_info<cl::sycl::info::device::max_work_item_sizes>();

//    sycl_dim_3_t launch_threads {sizes.get(0), sizes.get(1)/3, sizes.get(2)/3};
    sycl_dim_3_t launch_threads;
    launch_threads.x = threads.x;  //std::min(launch_threads.x, threads.x); 
    launch_threads.y = threads.y;  //std::min(launch_threads.y, threads.y);
    launch_threads.z = threads.z; //std::min(launch_threads.z, threads.z);
//    sycl_dim_3_t launch_threads {256, 1, 1};

    sycl_dim_3_t launch_blocks;
    launch_blocks.x = launch_threads.x * blocks.x;
    launch_blocks.y = launch_threads.y * blocks.y;
    launch_blocks.z = launch_threads.z * blocks.z;

    sycl_dim_3_t launch_global;
    launch_global.x = launch_threads.x * ((global.x + (launch_threads.x - 1)) / launch_threads.x);
    launch_global.y = launch_threads.y * ((global.y + (launch_threads.y - 1)) / launch_threads.y);
    launch_global.z = launch_threads.z * ((global.z + (launch_threads.z - 1)) / launch_threads.z);

    std::cout << "\nGlobal.x = " << launch_global.x
              << "\nGlobal.y = " << launch_global.y
              << "\nGlobal.z = " << launch_global.z
              << "\nThreads.x = " << launch_threads.x
              << "\nThreads.y = " << launch_threads.y
              << "\nThreads.z = " << launch_threads.z;

    cl::sycl::range<3> ret_th = {launch_threads.x, launch_threads.y, launch_threads.z};
    cl::sycl::range<3> ret_gl = {launch_global.x, launch_global.y, launch_global.z};

    return cl::sycl::nd_range<3>(ret_gl, ret_th);
  }
};

template <camp::idx_t cur_stmt, camp::idx_t num_stmts, typename StmtList>
struct SyclStatementListExecutorHelper {

  using next_helper_t =
      SyclStatementListExecutorHelper<cur_stmt + 1, num_stmts, StmtList>;

  using cur_stmt_t = camp::at_v<StmtList, cur_stmt>;

  template <typename Data>
  inline static RAJA_DEVICE void exec(Data &data, cl::sycl::nd_item<3> item, bool thread_active)
  {
    // Execute stmt
    cur_stmt_t::exec(data, item, thread_active);

    // Execute next stmt
    next_helper_t::exec(data, item, thread_active);
  }

  template <typename Data>
  inline static LaunchDims calculateDimensions(Data &data)
  {
    // Compute this statements launch dimensions
    LaunchDims statement_dims = cur_stmt_t::calculateDimensions(data);

    // call the next statement in the list
    LaunchDims next_dims = next_helper_t::calculateDimensions(data);

    // Return the maximum of the two
    return statement_dims.max(next_dims);
  }
};

template <camp::idx_t num_stmts, typename StmtList>
struct SyclStatementListExecutorHelper<num_stmts, num_stmts, StmtList> {

  template <typename Data>
  inline static RAJA_DEVICE void exec(Data &, cl::sycl::nd_item<3> item, bool)
  {
    // nop terminator
  }

  template <typename Data>
  inline static LaunchDims calculateDimensions(Data &)
  {
    return LaunchDims();
  }
};

template <typename Data, typename Policy, typename Types>
struct SyclStatementExecutor;

template <typename Data, typename StmtList, typename Types>
struct SyclStatementListExecutor;


template <typename Data, typename... Stmts, typename Types>
struct SyclStatementListExecutor<Data, StatementList<Stmts...>, Types> {

  using enclosed_stmts_t =
      camp::list<SyclStatementExecutor<Data, Stmts, Types>...>;

  static constexpr size_t num_stmts = sizeof...(Stmts);

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, cl::sycl::nd_item<3> item, bool thread_active)
  {
    // Execute statements in order with helper class
    SyclStatementListExecutorHelper<0, num_stmts, enclosed_stmts_t>::exec(data, item, thread_active);
  }

  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    // Compute this statements launch dimensions
    return SyclStatementListExecutorHelper<0, num_stmts, enclosed_stmts_t>::
        calculateDimensions(data);
  }
};

template <typename StmtList, typename Data, typename Types>
using sycl_statement_list_executor_t = SyclStatementListExecutor<
    Data,
    StmtList,
    Types>;

}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_SYCL guard

#endif  // closing endif for header file include guard
