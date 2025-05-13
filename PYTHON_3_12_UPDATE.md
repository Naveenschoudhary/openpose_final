# OpenPose Python 3.12 Update

This update modifies OpenPose to add support for Python 3.12, updates the pybind11 submodule, adds C++14 standard support, and improves Caffe compatibility.

## Changes Made

### 1. Updated pybind11 Submodule

- Updated pybind11 from version 2.2.0 to 2.11.1
- This version supports Python 3.12 out of the box

### 2. C++14 Support

- Enforced C++14 standard across the codebase
- Added `CMAKE_CXX_STANDARD` variable for better cross-platform compatibility

### 3. Python Binding Updates

- Updated binding code for compatibility with newer pybind11 versions
- Fixed bindings to work with Python 3.12

### 4. Caffe Updates

- Changed default Python version from 2 to 3
- Updated Python dependency requirements for newer versions
- Added pybind11 support for Python 3.12+ in Caffe
- Created fallback to Boost.Python for older Python versions

## Building with Python 3.12

To build OpenPose with Python 3.12 support:

1. Make sure you have Python 3.12 installed on your system
2. Configure CMake with the following options:
   ```bash
   mkdir build
   cd build
   cmake -DBUILD_PYTHON=ON -DPYTHON_VERSION=3 ..
   make -j`nproc`
   ```

## Dependencies

Updated Python dependencies:

- Cython>=0.29.36
- numpy>=1.22.0
- scipy>=1.9.0
- scikit-image>=0.20.0
- matplotlib>=3.7.0
- ipython>=8.10.0
- h5py>=3.8.0
- And others as listed in `3rdparty/caffe/python/requirements.txt`

## Known Issues

If you encounter any issues with building OpenPose with Python 3.12 support, please check:

1. Make sure your CUDA driver version is compatible with your GPU
2. Ensure all Python dependencies are installed
3. If Caffe fails to build, try building with the included pybind11 version instead of system version

## References

- [pybind11 2.11.1 documentation](https://pybind11.readthedocs.io/en/stable/)
- [Python 3.12 release notes](https://docs.python.org/3/whatsnew/3.12.html)
