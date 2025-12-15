cd ../
mkdir build
cd build

cmake -DCMAKE_INSTALL_PREFIX=/work/lab/science_tokyo/fluid_sim_dealii/bin \
      -DCMAKE_BUILD_TYPE=Release \
      -DDIM_2D=ON \
      ..

make -j && make install
