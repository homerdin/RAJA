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
  unsigned long x, y, z;
};

using sycl_dim_t = cl::sycl::range<1>;

using sycl_dim_3_t = uint3;

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

}  // namespace sycl
}  // namespace policy

using policy::sycl::sycl_exec;

// TODO
template<int dim, int BLOCK_SIZE>
struct sycl_global_123{};

template<int BLOCK_SIZE>
using sycl_global_1 = sycl_global_123<0, BLOCK_SIZE>;
template<int BLOCK_SIZE>
using sycl_global_2 = sycl_global_123<1, BLOCK_SIZE>;
template<int BLOCK_SIZE>
using sycl_global_3 = sycl_global_123<2, BLOCK_SIZE>;

/*!
 * Maps segment indices to SYCL threads.
 * This is the lowest overhead mapping, but requires that there are enough
 * physical threads to fit all of the direct map requests.
 * For example, a segment of size 2000 will not fit, and trigger a runtime
 * error.
 */
template<int dim>
struct sycl_group_123_loop{};

using sycl_group_1_loop = sycl_group_123_loop<0>;
using sycl_group_2_loop = sycl_group_123_loop<1>;
using sycl_group_3_loop = sycl_group_123_loop<2>;

/*!
 * Maps segment indices to SYCL threads.
 * This is the lowest overhead mapping, but requires that there are enough
 * physical threads to fit all of the direct map requests.
 * For example, a segment of size 2000 will not fit, and trigger a runtime
 * error.
 */
template<int dim>
struct sycl_item_123_loop{};

using sycl_item_1_loop = sycl_item_123_loop<0>;
using sycl_item_2_loop = sycl_item_123_loop<1>;
using sycl_item_3_loop = sycl_item_123_loop<2>;

/*!
 * Maps segment indices to SYCL threads.
 * This is the lowest overhead mapping, but requires that there are enough
 * physical threads to fit all of the direct map requests.
 * For example, a segment of size 2000 will not fit, and trigger a runtime
 * error.
 */
template<int dim>
struct sycl_group_123_direct{};

using sycl_group_1_direct = sycl_group_123_direct<0>;
using sycl_group_2_direct = sycl_group_123_direct<1>;
using sycl_group_3_direct = sycl_group_123_direct<2>;

/*!
 * Maps segment indices to SYCL threads.
 * This is the lowest overhead mapping, but requires that there are enough
 * physical threads to fit all of the direct map requests.
 * For example, a segment of size 2000 will not fit, and trigger a runtime
 * error.
 */
template<int dim>
struct sycl_item_123_direct{};

using sycl_item_1_direct = sycl_item_123_direct<0>;
using sycl_item_2_direct = sycl_item_123_direct<1>;
using sycl_item_3_direct = sycl_item_123_direct<2>;

}  // namespace RAJA

#endif
