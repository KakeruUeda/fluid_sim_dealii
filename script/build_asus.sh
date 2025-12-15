cd ../
mkdir build
cd build

cmake -DCMAKE_INSTALL_PREFIX=/home/ku/dev/fluid_sim_dealii/bin \
      -DCMAKE_C_COMPILER=mpicc \
      -DCMAKE_CXX_COMPILER=mpicxx \
      -DCMAKE_BUILD_TYPE=Release \
      -DDIM_2D=ON \
      ..

make -j 
sudo make install

