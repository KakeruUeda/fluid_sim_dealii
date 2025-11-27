cd ../
mkdir build
cd build

cmake -D CMAKE_INSTALL_PREFIX=/work/lab/science_tokyo/fluid_sim_dealii/bin \
      -DCMAKE_BUILD_TYPE=Release \
      -DDIM_2D=ON \
      ..

make -j && make install
