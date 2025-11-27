cmake_minimum_required(VERSION 3.13.4)

set(TARGET "my_project")
project(${TARGET} C CXX)

# ----- set spatial dimension -----
option(DIM_2D "Compile with 2D" OFF)
option(DIM_3D "Compile with 3D" OFF)

if(DIM_2D)
  add_compile_definitions(SPATIAL_DIMENSION=2)
elseif(DIM_3D)
  add_compile_definitions(SPATIAL_DIMENSION=3)
else()
  message(FATAL_ERROR "You must set either -DDIM_2D=ON or -DDIM_3D=ON")
endif()

# ----- find dealii -----
find_package(deal.II 9.7.0 HINTS 
             ${deal.II_DIR} 
             ${DEAL_II_DIR} 
             ../ ../../ 
             $ENV{DEAL_II_DIR})

if (NOT deal.II_FOUND)
  message(FATAL_ERROR "Could not locate a recent deal.II.")
endif()

add_subdirectory(src)