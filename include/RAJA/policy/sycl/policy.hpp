/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA sequential policy definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef policy_sycl_HPP
#define policy_sycl_HPP

#include "RAJA/policy/PolicyBase.hpp"

#include <cstddef>

namespace RAJA
{

struct uint3 {
  unsigned int x, y, z;
};
//using sycl_dim_t = cl::sycl::range<1>;

using sycl_dim_t = uint3;

namespace detail
{
template <bool Async>
struct get_launch {
  static constexpr RAJA::Launch value = RAJA::Launch::async;
};

template <>
struct get_launch<false> {
  static constexpr RAJA::Launch value = RAJA::Launch::sync;
};
}  // end namespace detail

namespace policy
{
namespace sycl
{

//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////

template <size_t BLOCK_SIZE, bool Async = false>
struct sycl_exec : public RAJA::make_policy_pattern_launch_platform_t<
                       RAJA::Policy::sycl,
                       RAJA::Pattern::forall,
                       detail::get_launch<Async>::value,
                       RAJA::Platform::sycl> {
};


///
/// Segment execution policies
///

struct sycl_for_dynamic
    : make_policy_pattern_launch_platform_t<Policy::sycl,
                                            Pattern::forall,
                                            Launch::undefined,
                                            Platform::host> {
  std::size_t grain_size;
  sycl_for_dynamic(std::size_t grain_size_ = 1) : grain_size(grain_size_) {}
};


template <std::size_t GrainSize = 1>
struct sycl_for_static : make_policy_pattern_launch_platform_t<Policy::sycl,
                                                              Pattern::forall,
                                                              Launch::undefined,
                                                              Platform::host> {
};

using sycl_for_exec = sycl_for_static<>;

///
/// Index set segment iteration policies
///
using sycl_segit = sycl_for_exec;


///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///
/*struct sycl_reduce : make_policy_pattern_launch_platform_t<Policy::sycl,
                                                          Pattern::reduce,
                                                          Launch::undefined,
                                                          Platform::host> {
};
*/
}  // namespace sycl
}  // namespace policy

using policy::sycl::sycl_exec;
//using policy::sycl::sycl_for_dynamic;
//using policy::sycl::sycl_for_exec;
//using policy::sycl::sycl_for_static;
//using policy::sycl::sycl_reduce;
//using policy::sycl::sycl_segit;

/*!
 * Maps segment indices to SYCL threads.
 * This is the lowest overhead mapping, but requires that there are enough
 * physical threads to fit all of the direct map requests.
 * For example, a segment of size 2000 will not fit, and trigger a runtime
 * error.
 */
template<int dim>
struct sycl_group_123{};

using sycl_group_1 = sycl_group_123<0>;
using sycl_group_2 = sycl_group_123<1>;
using sycl_group_3 = sycl_group_123<2>;

/*!
 * Maps segment indices to SYCL threads.
 * This is the lowest overhead mapping, but requires that there are enough
 * physical threads to fit all of the direct map requests.
 * For example, a segment of size 2000 will not fit, and trigger a runtime
 * error.
 */
template<int dim>
struct sycl_item_123{};

using sycl_item_1 = sycl_item_123<0>;
using sycl_item_2 = sycl_item_123<1>;
using sycl_item_3 = sycl_item_123<2>;

/*using sycl_exec_1 = sycl_exec_1<1>;
using sycl_exec_2 = sycl_exec_2<1>;
using sycl_exec_3 = sycl_exec_3<1>;
*/
namespace internal{

} // namespace internal

}  // namespace RAJA

#endif
