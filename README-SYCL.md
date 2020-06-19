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

The overhead from the RAJA abstraction is being tested at:
https://github.com/homerdin/RAJAPerf raja-sycl branch

There is a build script for the Intel SYCL compiler at
scripts/alcf-builds/sycl.sh
