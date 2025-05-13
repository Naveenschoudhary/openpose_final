#!/bin/bash
# Script to install necessary BLAS libraries on Ubuntu

echo "Installing BLAS, LAPACK, and OpenBLAS libraries..."
apt-get update
apt-get install -y libopenblas-dev libatlas-base-dev liblapack-dev liblapacke-dev

# Create symbolic links if the specific cblas library isn't found
if [ ! -f /usr/lib/x86_64-linux-gnu/libcblas.so ]; then
  echo "Creating symbolic link for CBLAS library..."
  if [ -f /usr/lib/x86_64-linux-gnu/libopenblas.so ]; then
    ln -sf /usr/lib/x86_64-linux-gnu/libopenblas.so /usr/lib/x86_64-linux-gnu/libcblas.so
    echo "Created symbolic link from libopenblas.so to libcblas.so"
  elif [ -f /usr/lib/x86_64-linux-gnu/atlas/libcblas.so ]; then
    ln -sf /usr/lib/x86_64-linux-gnu/atlas/libcblas.so /usr/lib/x86_64-linux-gnu/libcblas.so
    echo "Created symbolic link from atlas libcblas.so to /usr/lib/libcblas.so"
  else
    echo "WARNING: Could not find a CBLAS library to link to"
  fi
fi

echo "BLAS libraries installed successfully."
echo "Now try building OpenPose with: cmake -DBUILD_PYTHON=ON -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=Auto -DBLAS=Open .." 