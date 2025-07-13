cmake_minimum_required(VERSION 3.13.4)

set(TARGET "my_project")
project(${TARGET} C CXX)

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