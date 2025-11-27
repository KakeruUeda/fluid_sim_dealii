# Fluid Simulation with deal.II

A computational fluid dynamics (CFD) solver built with the deal.II finite element library. This project implements stabilized finite element methods for solving Navier-Stokes equations.

## Features
- Support for both 2D and 3D 
- MPI parallelization support
- Custom boundary condition 

## Dependencies

- deal.II
- CMake
- MPI (OpenMPI, MPICH)
- HDF5 (IO)
- GMSH (mesh generator)

## Example
### Flow through a tube (Re = 2000)

**Velocity Field**

https://github.com/KakeruUeda/fluid_sim_dealii/raw/main/media/stenosis_re2000_velocity.mp4

**Vorticity Field**

https://github.com/KakeruUeda/fluid_sim_dealii/raw/main/media/stenosis_re2000_voritcity.mp4  


