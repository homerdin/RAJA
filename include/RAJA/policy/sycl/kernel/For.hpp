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

template <typename Data, typename Types, typename... EnclosedStmts>
struct SyclForWrapper : public GenericWrapperBase
{
  using data_t = camp::decay<Data>;

  data_t data;

  /*! 
 *    * \brief Deferences data so that it can be mapped to the device
 *       */
  RAJA_INLINE
  constexpr explicit SyclForWrapper(data_t &d) :
    data{d}  {}

  RAJA_INLINE
  void exec() { execute_statement_list<camp::list<EnclosedStmts...>, Types>(data); }

  template <typename InIndexType>
  RAJA_INLINE void operator()(InIndexType i)
  {
//    data.template assign_offset<ArgumentId>(i);
    exec();
  }
};

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
    statement::For<ArgumentId, RAJA::sycl_exec<BlockDim>, EnclosedStmts...>,
    Types> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      SyclStatementListExecutor<Data, stmt_list_t, NewTypes>;


  static
  inline RAJA_DEVICE void exec(Data &data, cl::sycl::nd_item<1> item)
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
    dims.threads = len;
    dims.blocks = 256 * ((len + 256 -1) / 256);
//    set_sycl_dim<BlockDim>(dims.blocks, len);

    // since we are direct-mapping, we REQUIRE len
//    set_sycl_dim<BlockDim>(dims.min_blocks, len);

    // combine with enclosed statements
    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);
    return dims.max(enclosed_dims);
  }
};

}  // namespace internal
}  // end namespace RAJA


#endif 
