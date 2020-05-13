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
template <size_t num_threads>
struct sycl_threadblock_exec
    : public make_policy_pattern_launch_platform_t<
          Policy::sycl,
          Pattern::forall,
          Launch::undefined,
          Platform::sycl> {
};

namespace internal
{

// LaunchDims and Helper functions
struct LaunchDims {
  sycl_dim_t blocks;
//  sycl_dim_t min_blocks;
  sycl_dim_t threads;
//  sycl_dim_t min_threads;

  RAJA_INLINE
  RAJA_HOST_DEVICE
  LaunchDims() : blocks{1,1,1}, // min_blocks{0},
                 threads{1,1,1} {}//, min_threads{0} {}

  RAJA_INLINE
  RAJA_HOST_DEVICE
  LaunchDims(LaunchDims const &c) :
  blocks(c.blocks),//   min_blocks(c.min_blocks),
  threads(c.threads)//, min_threads(c.min_threads)
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

/*    result.blocks = cl::sycl::range<1> {std::max(c.blocks.get(0), blocks.get(0))};

    result.min_blocks = cl::sycl::range<1> {std::max(c.min_blocks.get(0), min_blocks.get(0))};

    result.threads = cl::sycl::range<1> {std::max(c.threads.get(0), threads.get(0))};

    result.min_threads = cl::sycl::range<1> {std::max(c.min_threads.get(0), min_threads.get(0))};
*/
    return result;
  }
};

template <camp::idx_t cur_stmt, camp::idx_t num_stmts, typename StmtList>
struct SyclStatementListExecutorHelper {

  using next_helper_t =
      SyclStatementListExecutorHelper<cur_stmt + 1, num_stmts, StmtList>;

  using cur_stmt_t = camp::at_v<StmtList, cur_stmt>;
/*
  template <typename Data>
  inline static RAJA_DEVICE void exec(Data &data, cl::sycl::h_item<3> item)
  {
    // Execute stmt
    cur_stmt_t::exec(data, item);

    // Execute next stmt
    next_helper_t::exec(data, item);
  }
*/
  template <typename Data>
  inline static RAJA_DEVICE void exec(Data &data, cl::sycl::group<3> group, cl::sycl::h_item<3> item)
  {
    // Execute stmt
    cur_stmt_t::exec(data, group, item);

    // Execute next stmt
    next_helper_t::exec(data, group, item);
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
  inline static RAJA_DEVICE void exec(Data &, cl::sycl::group<3>, cl::sycl::h_item<3> item)
  {
    // nop terminator
  }
/*
  template <typename Data>
  inline static RAJA_DEVICE void exec(Data &, cl::sycl::h_item<3>)
  {
    // nop terminator
  }
*/
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
  void exec(Data &data, cl::sycl::group<3> group, cl::sycl::h_item<3> item)
  {
//    std::cout << "in SyclStatementListExecutor.exec, should call helper" << std::endl;
    // Execute statements in order with helper class
    SyclStatementListExecutorHelper<0, num_stmts, enclosed_stmts_t>::exec(data, group, item);
  }
/*
  static
  inline
  RAJA_DEVICE
  void exec(Data &data, cl::sycl::h_item<3> item)
  {
//    std::cout << "in SyclStatementListExecutor.exec, should call helper" << std::endl;
    // Execute statements in order with helper class
    SyclStatementListExecutorHelper<0, num_stmts, enclosed_stmts_t>::exec(data, item);
  }
*/
  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    // Compute this statements launch dimensions
    std::cout << "num_stmts = " << num_stmts << std::endl;
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
                                                          
