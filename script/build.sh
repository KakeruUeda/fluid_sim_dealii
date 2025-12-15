cd ../
mkdir build
cd build

cmake -DCMAKE_INSTALL_PREFIX=$PWD/example \
      -DCMAKE_C_COMPILER=mpicc \
      -DCMAKE_CXX_COMPILER=mpicxx \
      -DCMAKE_BUILD_TYPE=Release \
      -DDIM_2D=ON \
      ..

make -j 
make install

