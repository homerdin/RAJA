//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA workgroup unordered runs.
///

#ifndef __TEST_WORKGROUP_UNORDERED__
#define __TEST_WORKGROUP_UNORDERED__

#include "RAJA_test-workgroup.hpp"
#include "RAJA_test-forall-data.hpp"

#include <random>


template <typename T>
class WorkGroupBasicUnorderedFunctionalTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicUnorderedFunctionalTest);


template <typename ExecPolicy,
          typename OrderPolicy,
          typename StoragePolicy,
          typename IndexType,
          typename Allocator,
          typename WORKING_RES
          >
void testWorkGroupUnordered(IndexType begin, IndexType end)
{
  ASSERT_GE(begin, (IndexType)0);
  ASSERT_GE(end, begin);
  IndexType N = end + begin;

  camp::resources::Resource working_res{WORKING_RES()};

  IndexType* working_array;
  IndexType* check_array;
  IndexType* test_array;

  allocateForallTestData<IndexType>(N,
                                    working_res,
                                    &working_array,
                                    &check_array,
                                    &test_array);

  for (IndexType i = IndexType(0); i < N; i++) {
    test_array[i] = IndexType(0);
  }

  working_res.memcpy(working_array, test_array, sizeof(IndexType) * N);

  for (IndexType i = begin; i < end; ++i) {
    test_array[ i ] = IndexType(i) + IndexType(5);
  }

  RAJA::WorkPool<
                  RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                  IndexType,
                  RAJA::xargs<>,
                  Allocator
                >
      pool(Allocator{});

  pool.enqueue(RAJA::TypedRangeSegment<IndexType>{ begin, end },
      [=] RAJA_HOST_DEVICE (IndexType i) {
    working_array[i] += i + IndexType(5);
  });

  auto group = pool.instantiate();

  auto site = group.run();

  working_res.memcpy(check_array, working_array, sizeof(IndexType) * N);

  //
  for (IndexType i = IndexType(0); i < N; i++) {
    ASSERT_EQ(test_array[i], check_array[i]);
  }

  deallocateForallTestData<IndexType>(working_res,
                                      working_array,
                                      check_array,
                                      test_array);
}

TYPED_TEST_P(WorkGroupBasicUnorderedFunctionalTest, BasicWorkGroupUnordered)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<3>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<4>>::type;
  using WORKING_RESOURCE = typename camp::at<TypeParam, camp::num<5>>::type;

  std::mt19937 rng(std::random_device{}());
  using dist_type = std::uniform_int_distribution<IndexType>;

  IndexType b1 = dist_type(IndexType(0), IndexType(15))(rng);
  IndexType e1 = dist_type(b1, IndexType(16))(rng);

  IndexType b2 = dist_type(e1, IndexType(127))(rng);
  IndexType e2 = dist_type(b2, IndexType(128))(rng);

  IndexType b3 = dist_type(e2, IndexType(1023))(rng);
  IndexType e3 = dist_type(b3, IndexType(1024))(rng);

  testWorkGroupUnordered< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator, WORKING_RESOURCE >(b1, e1);
  testWorkGroupUnordered< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator, WORKING_RESOURCE >(b2, e2);
  testWorkGroupUnordered< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator, WORKING_RESOURCE >(b3, e3);
}


template <typename ExecPolicy,
          typename OrderPolicy,
          typename StoragePolicy,
          typename IndexType,
          typename Allocator,
          typename WORKING_RES
          >
void testWorkGroupUnorderedMultiple(
    IndexType begin, IndexType end,
    IndexType num1, IndexType num2, IndexType num3)
{
  ASSERT_GE(begin, (IndexType)0);
  ASSERT_GE(end, begin);
  IndexType N = end + begin;

  camp::resources::Resource working_res{WORKING_RES()};

  using type1 = IndexType;
  using type2 = size_t;
  using type3 = double;

  type1* working_array1 = nullptr;
  type1* check_array1 = nullptr;
  type1* test_array1 = nullptr;

  type2* working_array2 = nullptr;
  type2* check_array2 = nullptr;
  type2* test_array2 = nullptr;

  type3* working_array3 = nullptr;
  type3* check_array3 = nullptr;
  type3* test_array3 = nullptr;

  allocateForallTestData<type1>(N * num1,
                                working_res,
                                &working_array1,
                                &check_array1,
                                &test_array1);

  allocateForallTestData<type2>(N * num2,
                                working_res,
                                &working_array2,
                                &check_array2,
                                &test_array2);

  allocateForallTestData<type3>(N * num3,
                                working_res,
                                &working_array3,
                                &check_array3,
                                &test_array3);


  for (IndexType j = IndexType(0); j < num1; j++) {
    type1* test_ptr1 = test_array1 + N * j;
    for (IndexType i = IndexType(0); i < N; i++) {
      test_ptr1[i] = type1(0);
    }
  }

  for (IndexType j = IndexType(0); j < num2; j++) {
    type2* test_ptr2 = test_array2 + N * j;
    for (IndexType i = IndexType(0); i < N; i++) {
      test_ptr2[i] = type2(0);
    }
  }

  for (IndexType j = IndexType(0); j < num3; j++) {
    type3* test_ptr3 = test_array3 + N * j;
    for (IndexType i = IndexType(0); i < N; i++) {
      test_ptr3[i] = type3(0);
    }
  }


  working_res.memcpy(working_array1, test_array1, sizeof(type1) * N * num1);

  working_res.memcpy(working_array2, test_array2, sizeof(type2) * N * num2);

  working_res.memcpy(working_array3, test_array3, sizeof(type3) * N * num3);


  for (IndexType j = IndexType(0); j < num1; j++) {
    type1* test_ptr1 = test_array1 + N * j;
    for (IndexType i = begin; i < end; ++i) {
      test_ptr1[ i ] = type1(i) + type1(5);
    }
  }

  for (IndexType j = IndexType(0); j < num2; j++) {
    type2* test_ptr2 = test_array2 + N * j;
    for (IndexType i = begin; i < end; ++i) {
      test_ptr2[ i ] = type2(i) + type2(7);
    }
  }

  for (IndexType j = IndexType(0); j < num3; j++) {
    type3* test_ptr3 = test_array3 + N * j;
    for (IndexType i = begin; i < end; ++i) {
      test_ptr3[ i ] = type3(i) + type3(11);
    }
  }


  RAJA::WorkPool<
                  RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                  IndexType,
                  RAJA::xargs<>,
                  Allocator
                >
      pool(Allocator{});

  for (IndexType j = IndexType(0); j < num1; j++) {
    type1* working_ptr1 = working_array1 + N * j;
    pool.enqueue(RAJA::TypedRangeSegment<IndexType>{ begin, end },
        [=] RAJA_HOST_DEVICE (IndexType i) {
      working_ptr1[i] += type1(i) + type1(5);
    });
  }

  for (IndexType j = IndexType(0); j < num2; j++) {
    type2* working_ptr2 = working_array2 + N * j;
    pool.enqueue(RAJA::TypedRangeSegment<IndexType>{ begin, end },
        [=] RAJA_HOST_DEVICE (IndexType i) {
      working_ptr2[i] += type2(i) + type2(7);
    });
  }

  for (IndexType j = IndexType(0); j < num3; j++) {
    type3* working_ptr3 = working_array3 + N * j;
    pool.enqueue(RAJA::TypedRangeSegment<IndexType>{ begin, end },
        [=] RAJA_HOST_DEVICE (IndexType i) {
      working_ptr3[i] += type3(i) + type3(11);
    });
  }

  auto group = pool.instantiate();

  auto site = group.run();


  working_res.memcpy(check_array1, working_array1, sizeof(type1) * N * num1);

  working_res.memcpy(check_array2, working_array2, sizeof(type2) * N * num2);

  working_res.memcpy(check_array3, working_array3, sizeof(type3) * N * num3);


  for (IndexType j = IndexType(0); j < num1; j++) {
    type1* test_ptr1 = test_array1 + N * j;
    type1* check_ptr1 = check_array1 + N * j;
    for (IndexType i = IndexType(0); i < N; i++) {
      ASSERT_EQ(test_ptr1[i], check_ptr1[i]);
    }
  }

  for (IndexType j = IndexType(0); j < num2; j++) {
    type2* test_ptr2 = test_array2 + N * j;
    type2* check_ptr2 = check_array2 + N * j;
    for (IndexType i = IndexType(0); i < N; i++) {
      ASSERT_EQ(test_ptr2[i], check_ptr2[i]);
    }
  }

  for (IndexType j = IndexType(0); j < num3; j++) {
    type3* test_ptr3 = test_array3 + N * j;
    type3* check_ptr3 = check_array3 + N * j;
    for (IndexType i = IndexType(0); i < N; i++) {
      ASSERT_EQ(test_ptr3[i], check_ptr3[i]);
    }
  }


  deallocateForallTestData<type1>(working_res,
                                  working_array1,
                                  check_array1,
                                  test_array1);

  deallocateForallTestData<type2>(working_res,
                                  working_array2,
                                  check_array2,
                                  test_array2);

  deallocateForallTestData<type3>(working_res,
                                  working_array3,
                                  check_array3,
                                  test_array3);
}

TYPED_TEST_P(WorkGroupBasicUnorderedFunctionalTest, BasicWorkGroupUnorderedMultiple)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<3>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<4>>::type;
  using WORKING_RESOURCE = typename camp::at<TypeParam, camp::num<5>>::type;

  std::mt19937 rng(std::random_device{}());
  using dist_type = std::uniform_int_distribution<IndexType>;

  IndexType begin = dist_type(IndexType(1), IndexType(16383))(rng);
  IndexType end   = dist_type(begin,        IndexType(16384))(rng);

  IndexType num1 = dist_type(IndexType(0), IndexType(32))(rng);
  IndexType num2 = dist_type(IndexType(0), IndexType(32))(rng);
  IndexType num3 = dist_type(IndexType(0), IndexType(32))(rng);

  testWorkGroupUnorderedMultiple< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator, WORKING_RESOURCE >(
      begin, end, num1, num2, num3);
}


REGISTER_TYPED_TEST_SUITE_P(WorkGroupBasicUnorderedFunctionalTest,
                            BasicWorkGroupUnordered,
                            BasicWorkGroupUnorderedMultiple);

#endif  //__TEST_WORKGROUP_UNORDERED__
