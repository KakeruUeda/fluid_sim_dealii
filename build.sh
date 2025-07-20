mkdir build
cd build

cmake -D CMAKE_INSTALL_PREFIX=/work/lab/fluid_sim_dealii/bin \
      -DCMAKE_BUILD_TYPE=Release \
      -DDIM_3D=ON \
      ..

make -j && make install