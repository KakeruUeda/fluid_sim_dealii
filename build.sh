#!/bin/sh
mkdir build
cd build
cmake -D CMAKE_INSTALL_PREFIX=/work/lab/fluid_sim_dealii/bin \
      ..

make && make install