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
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          int BlockDim,
          typename... EnclosedStmts,
          typename Types>
struct SyclStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::sycl_group_123<BlockDim>, EnclosedStmts...>,
    Types> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      SyclStatementListExecutor<Data, stmt_list_t, NewTypes>;


  static
  inline RAJA_DEVICE void exec(Data &data, cl::sycl::group<3> group, cl::sycl::h_item<3> item)
  {
    auto len = segment_length<ArgumentId>(data);
    auto i = group.get_id(BlockDim);
    if (i < len) {

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, group, item);
    }
  }


  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    auto len = segment_length<ArgumentId>(data);

    // request one block per element in the segment
    LaunchDims dims;
    if (BlockDim == 0) {
//      dims.threads.x = BlockSize;
//      dims.blocks.x = BlockSize * ((len + (BlockSize - 1)) / BlockSize);
      dims.blocks.x = len;
    } 
    if (BlockDim == 1) {
//      dims.threads.y = BlockSize;
//      dims.blocks.y = BlockSize * ((len + (BlockSize - 1)) / BlockSize);;
      dims.blocks.y = len;
    }
    if (BlockDim == 2) {
//      dims.threads.z = BlockSize;
//      dims.blocks.z = BlockSize * ((len + (BlockSize - 1)) / BlockSize);;
      dims.blocks.y = len;
    }
//    dims.blocks = 256 * ((len + 256 -1) / 256);
//    set_sycl_dim<BlockDim>(len);

    // since we are direct-mapping, we REQUIRE len
//    set_sycl_dim<BlockDim>(dims.min_blocks, len);

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
          int BlockDim,
          typename... EnclosedStmts,
          typename Types>
struct SyclStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::sycl_item_123<BlockDim>, EnclosedStmts...>,
    Types> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      SyclStatementListExecutor<Data, stmt_list_t, NewTypes>;


  static
  inline RAJA_DEVICE void exec(Data &data, cl::sycl::group<3> group, cl::sycl::h_item<3> item)
  {
//    group.parallel_for_work_item([&] (cl::sycl::h_item<3> item) {
    auto len = segment_length<ArgumentId>(data);
    auto i = item.get_local_id(BlockDim);
    if (i < len) {
      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, group, item);
    }
 //   });
  }

  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    auto len = segment_length<ArgumentId>(data);

    // request one block per element in the segment
    LaunchDims dims;
    if (BlockDim == 0) {
//      dims.threads.x = BlockSize;
//      dims.blocks.x = BlockSize * ((len + (BlockSize - 1)) / BlockSize);
      dims.threads.x = len;
    }
    if (BlockDim == 1) {
//      dims.threads.y = BlockSize;
//      dims.blocks.y = BlockSize * ((len + (BlockSize - 1)) / BlockSize);;
      dims.threads.y = len;
    }
    if (BlockDim == 2) {
//      dims.threads.z = BlockSize;
//      dims.blocks.z = BlockSize * ((len + (BlockSize - 1)) / BlockSize);;
      dims.threads.y = len;
    }
//    dims.blocks = 256 * ((len + 256 -1) / 256);
//    set_sycl_dim<BlockDim>(len);

    // since we are direct-mapping, we REQUIRE len
//    set_sycl_dim<BlockDim>(dims.min_blocks, len);

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
          int BlockDim,
          typename... EnclosedStmts,
          typename Types>
struct SyclStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::sycl_exec<BlockDim>, EnclosedStmts...>,
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

/*  static
  inline
  RAJA_DEVICE
  void exec(Data &data, cl::sycl::h_item<3> item)
  {

    using idx_type = camp::decay<decltype(camp::get<ArgumentId>(data.offset_tuple))>;

    idx_type len = segment_length<ArgumentId>(data);

    for(idx_type i = 0;i < len;++ i){
      // Assign i to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, item);
    }
  }
*/
  static
  inline
  RAJA_DEVICE
  void exec(Data &data, cl::sycl::group<3> group, cl::sycl::h_item<3> item)
  {

    using idx_type = camp::decay<decltype(camp::get<ArgumentId>(data.offset_tuple))>;

    idx_type len = segment_length<ArgumentId>(data);

    for(idx_type i = 0;i < len;++ i){
      // Assign i to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, group, item);
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
