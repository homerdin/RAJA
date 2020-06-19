# SYCL

This branch contains a WIP port of RAJAPerf to SYCL using the Intel
USM extension.

It is built against the branch of CAMP at:
https://github.com/homerdin/camp sycl branch

Some of the changes in RAJA and CAMP are to allow for trivially
copyable classes/structs in the RAJA-SYCL kernels.  Views are 
still not fully trivially copyable.

A setQueue(cl::sycl::queue&) function has been added to pass the 
sycl queue to be used by RAJA for execution. The USM extension 
from Intel is tied to a context and device.  Without using the 
application queue, device memory cannot be accessed.

Policies:
forall = sycl_exec<BLOCK_SIZE, async = true>

SyclKernel policies
sycl_global_1<BLOCK_SIZE> -> maps to nd_item.get_global_id(dim1)
sycl_global_2<BLOCK_SIZE> -> maps to nd_item.get_global_id(dim2)
sycl_global_3<BLOCK_SIZE> -> maps to nd_item.get_global_id(dim3)

sycl_group_1_loop
sycl_group_2_loop
sycl_group_3_loop

sycl_group_1_direct -> maps to nd_item.get_group(dim1)
sycl_group_2_direct -> maps to nd_item.get_group(dim1)
sycl_group_3_direct -> maps to nd_item.get_group(dim1)

sycl_local_1_loop
sycl_local_2_loop
sycl_local_3_loop

sycl_local_1_direct -> maps to nd_item.get_local_id(dim1)
sycl_local_2_direct -> maps to nd_item.get_local_id(dim2)
sycl_local_3_direct -> maps to nd_item.get_local_id(dim3)

The overhead from the RAJA abstraction is being tested at:
https://github.com/homerdin/RAJAPerf raja-sycl branch

There is a build script for the Intel SYCL compiler at
scripts/alcf-builds/sycl.sh
