# cumlprims OPG GoogleTests

Running the Googletests requires a CUDA-aware MPI version to be installed.

The following command will execute the cumlprims multi-GPU Googletests `mpirun  -np 2 --mca btl_smcuda_use_cuda_ipc 0 mlcommon_test_opg`.

These *should* be able to run multi-node as well, assuming this has been configured on your system.
