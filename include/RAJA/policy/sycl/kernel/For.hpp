/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for SYCL statement executors.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_sycl_kernel_For_HPP
#define RAJA_policy_sycl_kernel_For_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/sycl/kernel/internal.hpp"


namespace RAJA
{

namespace internal
{

// SyclStatementExecutors
//

/*
 * Executor for block work sharing inside SyclKernel.
 * Mapping directly to indicies
 * Assigns the global index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          int Dim,
          int Block_Size,
          typename... EnclosedStmts,
          typename Types>
struct SyclStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::sycl_global_123<Dim, Block_Size>, EnclosedStmts...>,
    Types> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      SyclStatementListExecutor<Data, stmt_list_t, NewTypes>;


  static
  inline RAJA_DEVICE void exec(Data &data, cl::sycl::nd_item<3> item, bool thread_active)
  {
    auto len = segment_length<ArgumentId>(data);
    auto i = item.get_global_id(Dim);

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, item, thread_active && (i<len));
  }

  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    auto len = segment_length<ArgumentId>(data);

    // request one block per element in the segment
    LaunchDims dims;
    if (Dim == 0) {
      dims.global.x = len;
      dims.threads.x = Block_Size;
    }
    if (Dim == 1) {
      dims.global.y = len;
      dims.threads.y = Block_Size;
    }
    if (Dim == 2) {
      dims.global.z = len;
      dims.threads.z = Block_Size;
    }

    // combine with enclosed statements
    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);
    return dims.max(enclosed_dims);
  }
};

/*
 * Executor for block work sharing inside SyclKernel.
 * Mapping directly to indicies
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          int Dim,
          typename... EnclosedStmts,
          typename Types>
struct SyclStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::sycl_group_123_direct<Dim>, EnclosedStmts...>,
    Types> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      SyclStatementListExecutor<Data, stmt_list_t, NewTypes>;


  static
  inline RAJA_DEVICE void exec(Data &data, cl::sycl::nd_item<3> item, bool thread_active)
  {
    auto len = segment_length<ArgumentId>(data);
    auto i = item.get_group(Dim);

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, item, thread_active && (i<len));
  }

  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    auto len = segment_length<ArgumentId>(data);

    // request one block per element in the segment
    LaunchDims dims;
    if (Dim == 0) {
      dims.blocks.x = len;
    }
    if (Dim == 1) {
      dims.blocks.y = len;
    }
    if (Dim == 2) {
      dims.blocks.z = len;
    }

    // combine with enclosed statements
    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);
    return dims.max(enclosed_dims);
  }
};

/*
 * Executor for block work sharing inside CudaKernel.
 * Provides a group-stride loop (stride of grid range) for
 * each group in dims.
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          int Dim,
          typename... EnclosedStmts,
          typename Types>
struct SyclStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::sycl_group_123_loop<Dim>, EnclosedStmts...>,
    Types> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      SyclStatementListExecutor<Data, stmt_list_t, NewTypes>;


  static
  inline RAJA_DEVICE void exec(Data &data, cl::sycl::nd_item<3> item, bool thread_active)
  {
    auto len = segment_length<ArgumentId>(data);
    auto i0 = item.get_group(Dim);
    auto i_stride = item.get_group_range(Dim);

    for(auto i = i0;i < len;i += i_stride){

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, item, thread_active);
    }
  }

  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    auto len = segment_length<ArgumentId>(data);

    // request one block per element in the segment
    LaunchDims dims;
    if (Dim == 0) {
      dims.blocks.x = len;
    } 
    if (Dim == 1) {
      dims.blocks.y = len;
    }
    if (Dim == 2) {
      dims.blocks.z = len;
    }

    // combine with enclosed statements
    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);
    return dims.max(enclosed_dims);
  }
};

/*
 * Executor for block work sharing inside SyclKernel.
 * Mapping directly to indicies
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          int Dim,
          typename... EnclosedStmts,
          typename Types>
struct SyclStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::sycl_local_123_direct<Dim>, EnclosedStmts...>,
    Types> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      SyclStatementListExecutor<Data, stmt_list_t, NewTypes>;


  static
  inline RAJA_DEVICE void exec(Data &data, cl::sycl::nd_item<3> item, bool thread_active)
  {
    auto len = segment_length<ArgumentId>(data);
    auto i = item.get_local_id(Dim);

    // assign thread id directly to offset
    data.template assign_offset<ArgumentId>(i);

    // execute enclosed statements if in bounds
    enclosed_stmts_t::exec(data, item, thread_active && (i<len));

  }

  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    auto len = segment_length<ArgumentId>(data);

    // request one block per element in the segment
    LaunchDims dims;
    if (Dim == 0) {
      dims.threads.x = len;
    }
    if (Dim == 1) {
      dims.threads.y = len;
    }
    if (Dim == 2) {
      dims.threads.z = len;
    }

    // combine with enclosed statements
    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);
    return dims.max(enclosed_dims);
  }
};

/*
 * Executor for work item sharing loop inside CudaKernel.
 * Provides a local-stride loop (stride of work item local range)
 * for each item in dim.
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          int Dim,
          typename... EnclosedStmts,
          typename Types>
struct SyclStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::sycl_local_123_loop<Dim>, EnclosedStmts...>,
    Types> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      SyclStatementListExecutor<Data, stmt_list_t, NewTypes>;


  static
  inline RAJA_DEVICE void exec(Data &data, cl::sycl::nd_item<3> item, bool thread_active)
  {
    auto len = segment_length<ArgumentId>(data);
    auto i0 = item.get_local_id(Dim);
    auto i_stride = item.get_local_range(Dim);
    auto i = i0;

    for(; i < len;i += i_stride){

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, item, thread_active);
    }
    // do we need one more masked iteration?
    if(i - i0 < len)
    {
      // execute enclosed statements one more time, but masking them off
      // this is because there's at least one thread that isn't masked off
      // that is still executing the above loop
      enclosed_stmts_t::exec(data, item, false);
    }
  }

  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    auto len = segment_length<ArgumentId>(data);

// TODO
    // request one thread per element in the segment
//    LaunchDims dims;
//    set_cuda_dim<ThreadDim>(dims.threads, len);

    // request one block per element in the segment
    LaunchDims dims;
    if (Dim == 0) {
      dims.threads.x = len;
    }
    if (Dim == 1) {
      dims.threads.y = len;
    }
    if (Dim == 2) {
      dims.threads.z = len;
    }

    // combine with enclosed statements
    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);
    return dims.max(enclosed_dims);
  }
};


/*
 * Executor for block work sharing inside SyclKernel.
 * Mapping directly to indicies
 * Assigns the loop index to offset ArgumentId
 */
/*template <typename Data,
          camp::idx_t ArgumentId,
          int Dim,
          typename... EnclosedStmts,
          typename Types>
struct SyclStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::sycl_exec<Dim>, EnclosedStmts...>,
    Types> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      SyclStatementListExecutor<Data, stmt_list_t, NewTypes>;


  static
  inline RAJA_DEVICE void exec(Data &data, cl::sycl::nd_item<3> item)
  {
    auto len = segment_length<ArgumentId>(data);
    auto i = item.get_global_id(0);

    if (i < len) {

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, item);
    }
  }


  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    auto len = segment_length<ArgumentId>(data);

    // request one block per element in the segment
    LaunchDims dims;
    dims.threads.x = 1;
    dims.blocks.x = len;// 256 * ((len + 256 -1) / 256);
//    set_sycl_dim<BlockDim>(dims.blocks, len);

    // since we are direct-mapping, we REQUIRE len
  //  set_sycl_dim<BlockDim>(dims.min_blocks, len);

    // combine with enclosed statements
    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);
    return dims.max(enclosed_dims);
  }
};
*/
/*
 * Executor for sequential loops inside of a SyclKernel.
 *
 * This is specialized since it need to execute the loop immediately.
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename... EnclosedStmts,
          typename Types>
struct SyclStatementExecutor<
    Data,
    statement::For<ArgumentId, seq_exec, EnclosedStmts...>,
    Types> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      SyclStatementListExecutor<Data, stmt_list_t, NewTypes>;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, cl::sycl::nd_item<3> item, bool thread_active)
  {

    using idx_type = camp::decay<decltype(camp::get<ArgumentId>(data.offset_tuple))>;

    idx_type len = segment_length<ArgumentId>(data);

    for(idx_type i = 0;i < len;++ i){
      // Assign i to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, item, thread_active);
    }
  }

  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    return enclosed_stmts_t::calculateDimensions(data);
  }
};


}  // namespace internal
}  // end namespace RAJA


#endif 
