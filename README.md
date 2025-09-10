# cuMLPrims

This repository contains  C++ and CUDA code of muti-node multi-GPU (MNMG) ML mathematical primitives and some algorithms, that are used by [the main cuML project](https://github.com/rapidsai/cuml). The build system uses CMake for build configuration, and an out-of-source build is recommended.

As of version 0.13, the MNMG code included in cuMLPrims follows the model one-process-per-GPU (OPG), where the code uses a communication library (based on cuML's comms) and each process has one GPU assigned to it. This is in contrast to single-process-multi-GPU (SPMG) approaches, which are no longer part of the code base.

## Folder Structure

The folder structure mirrors closely the structure of github cuML. The folders are:

- `ci`: Folders containing CI related scripts to run tests for each MR and create the conda packages.
- `conda` Contains Conda recipe for `libcumlprims` Conda package in the `rapidsai` channel.
- `cpp`: Contains the source code.
    - `cpp/cmake`: CMake related scripts.
    - `cpp/include`: The include folder for headers that are necessary to be installed/distributed to use the libcumlprims.so artifact by users of the library.
    - `cpp/src_prims_opg`: Contains source code for MNMG ML primitives. It also contains source code for algorithms that use the primitives that are still included in cuMLPrims as opposed to cuML.
    - `cpp/test`: Googletest based unit tests (work in progress).

## Building cuMLPrims:

### Requirements

The main artifact produced by the build system is the shared library libcumlprims. Ensure the following dependencies are satisfied:

1. CMake (>= 3.14.5)
2. CUDA (>= 10.0)
3. GCC (>= 5.4.0)
4. NCCL (>= 2.4.6.1)
5. UCX with CUDA support (optional)(>=1.7) - enables point-to-point messaging in the cuML communicator.
6. zlib

It is recommended to use conda for environment/package management. If doing so, it is recommended to use the convenience environment .yml file located in [**the cuML repository**, in `conda/environments/cuml_dev_cudax.y.yml` (replace x.y for your CUDA version)](https://github.com/rapidsai/cuml/tree/branch-0.14/conda/environments). This file contains most of the dependencies ment1ioned above (notable exceptions are gcc and zlib). To use it, for example to create an environment named cuml_dev for CUDA 10.0 and Python 3.7, you can use the follow command:

```bash
conda env create -n cuml_dev python=3.7 --file=conda/environments/cuml_dev_cuda10.0.yml
```

### Using build.sh script

As a convenience, a `build.sh` script is provided which can be used to execute the build commands in an automated manner.  Note that the libraries will be installed to the location set in `$INSTALL_PREFIX` if set (i.e. `export INSTALL_PREFIX=/install/path`), otherwise to `$CONDA_PREFIX`.
```bash
$ ./build.sh                           # build the libcuml library and tests
                                       # install them to $INSTALL_PREFIX if set, otherwise $CONDA_PREFIX
```

Other `build.sh` options:

```bash
$ ./build.sh clean                     # remove any prior build artifacts and configuration (start over)
$ ./build.sh libcumlprims -v                # build and install libcumlprims with verbose output
$ ./build.sh libcumlprims -g                # build and install libcumlprims for debug
$ PARALLEL_LEVEL=4 ./build.sh libcumlprims  # build and install libcumlprims limiting parallel build jobs to 4 (make -j4)
$ ./build.sh libcuml -n                # build libcuml but do not install
$ ./build.sh libcumlprims --allgpuarch        # build the  tests for all supported GPU architectures

```

### General Build Procedure:

Once dependencies are present, follow the steps below:

1. Clone the repository (no need to use submodule functionality).


2. Configure the build of `libcumlprims` (C++/CUDA library containing the cuML algorithms), starting from the repository root folder:
```bash
$ cd cpp
$ mkdir build && cd build
$ export CUDA_BIN_PATH=$CUDA_HOME # (optional env variable if cuda binary is not in the PATH. Default CUDA_HOME=/path/to/cuda/)
$ cmake ..
```

To build tests the option `-DBUILD_OPG_TESTS=ON` needs to be passed, for more information see step 4.

If using a conda environment (recommended), then cmake can be configured appropriately for `libcuml++` via:

```bash
$ cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
```

Note: The following warning message is dependent upon the version of cmake and the `CMAKE_INSTALL_PREFIX` used. If this warning is displayed, the build should still run succesfully. We are currently working to resolve this open issue. You can silence this warning by adding `-DCMAKE_IGNORE_PATH=$CONDA_PREFIX/lib` to your `cmake` command.
```
Cannot generate a safe runtime search path for target ml_test because files
in some directories may conflict with libraries in implicit directories:
```

There are many options to configure the build process, see the [customizing build section](#libcuml-&-libcumlc++).

3. Build `libcumlprims`:

```bash
$ make -j
$ make install
```

4. Tests are optional currently, and can currently be run with CUDA aware MPI installed only. They are meant for development mainly currently (will be enabled in CI in the future). To test full end-to-end functionality of the `libcumlprims` package, the pytests of main cuML need to be run.

To do that, first build (and install) `libcumlprims`, and then refer to [cuML's build guide](https://github.com/rapidsai/cuml/blob/branch-0.14/BUILD.md). After building `libcuml++` and the python package `cuml`, the pytests under `python/cuml/test/dask` will run unit tests of the algorithms that use `libcumlprims`.

### Code Style

Install `pre-commit` and execute `pre-commit run --all-files` to run style checks for this repository, such as `clang-format`.

### Custom Build Options

cuMLPrims CMake has the following configurable flags available:

| Flag | Possible Values | Default Value | Behavior |
| --- | --- | --- | --- |
| BUILD_OPG_TESTS | [ON, OFF] | OFF | Build MPI cumlcomms based C++ unit tests (in progress, refer to step 4. of the build steps).  |
| BUILD_CUMLPRIMS_LIBRARY | [ON, OFF]  | ON  | Enable/disable building libcumprims shared library. |
| CMAKE_CXX11_ABI | [ON, OFF]  | ON  | Enable/disable the GLIBCXX11 ABI  |
| DISABLE_OPENMP | [ON, OFF]  | OFF  | Set to `ON` to disable OpenMP  |
| KERNEL_INFO | [ON, OFF]  | OFF  | Enable/disable kernel resource usage info in nvcc. |
| LINE_INFO | [ON, OFF]  | OFF  | Enable/disable lineinfo in nvcc.  |
| NVTX | [ON, OFF]  | OFF  | Enable/disable nvtx markers in libcumlprims.  |
| GPU_ARCHS |  List of GPU architectures, semicolon-separated | 60;70;75  | List of GPU architectures that all artifacts are compiled for.  |
