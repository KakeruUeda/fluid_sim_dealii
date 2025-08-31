#pragma once
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

template <int dim>
void print_mesh_info(const Triangulation<dim>& triangulation, 
                     ConditionalOStream& pcout)
{
  pcout << "Mesh info:" << std::endl
        << " dimension: " << dim << std::endl
        << " no. of cells: " << triangulation.n_active_cells() << std::endl;

  {
    std::map<types::boundary_id, unsigned int> boundary_count;
    for (const auto& face : triangulation.active_face_iterators())
      if (face->at_boundary()) boundary_count[face->boundary_id()]++;

    pcout << " boundary indicators: ";
    for (const auto& pair : boundary_count)
    {
      pcout << pair.first << '(' << pair.second << " times) ";
    }
    pcout << std::endl;
  }
}
