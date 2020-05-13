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

#ifndef RAJA_policy_sycl_kernel_Lambda_HPP
#define RAJA_policy_sycl_kernel_Lambda_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <cassert>
#include <climits>

#include "camp/camp.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel.hpp"
#include "RAJA/pattern/kernel/Lambda.hpp"


namespace RAJA
{
namespace internal
{

// SyclStatementExecutor for actually invoking the lambda

template <typename Data, camp::idx_t LambdaIndex, typename... Args, typename Types>
struct SyclStatementExecutor<Data, statement::Lambda<LambdaIndex, Args...>, Types> {

  static
  inline RAJA_DEVICE void exec(Data &data, cl::sycl::group<3> group, cl::sycl::h_item<3> item)
  {
    // Only execute the lambda if it hasn't been masked off
//    if(thread_active){
      StatementExecutor<statement::Lambda<LambdaIndex, Args...>, Types>::exec(data);
//    }

  }
/*
  static
  inline RAJA_DEVICE void exec(Data &data, cl::sycl::h_item<3> item)
  {

    // Only execute the lambda if it hasn't been masked off
//    if(thread_active) {
      StatementExecutor<statement::Lambda<LambdaIndex, Args...>, Types>::exec(data);
//    }

  }
*/
  static
  inline
  LaunchDims calculateDimensions(Data const & RAJA_UNUSED_ARG(data))
  {
    return LaunchDims();
  }
};


//
/*
template <typename Data, camp::idx_t LambdaIndex, typename... Args>
struct SyclStatementExecutor<Data, statement::Lambda<LambdaIndex, Args...>> {

  static
  inline RAJA_DEVICE void exec(Data &data, bool thread_active)
  {

    //Convert SegList, ParamList into Seg, Param types, and store in a list
    using targList = typename camp::flatten<camp::list<Args...>>::type;

    // Only execute the lambda if it hasn't been masked off
    if(thread_active){
      invoke_lambda_with_args<LambdaIndex, targList>(data);
    }

  }


  static
  inline
  LaunchDims calculateDimensions(Data const & RAJA_UNUSED_ARG(data))
  {
    return LaunchDims();
  }
};
*/

}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_SYCL guard

#endif  // closing endif for header file include guard
