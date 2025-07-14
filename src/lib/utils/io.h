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


template <int dim>
void output_results(std::string output_dir,
                    const Triangulation<dim>& triangulation, 
                    DoFHandler<dim> &dof_handler, 
                    PETScWrappers::MPI::Vector &solution,
                    MPI_Comm &mpi_comm,
                    unsigned int step)
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  std::vector<std::string> solution_names(dim, "velocity");
  solution_names.emplace_back("pressure");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  interpretation.push_back(DataComponentInterpretation::component_is_scalar);

  data_out.add_data_vector(
    solution, solution_names,
    DataOut<dim>::type_dof_data,
    interpretation);

  std::vector<unsigned int> partition_int(triangulation.n_active_cells());
  GridTools::get_subdomain_association(triangulation, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  DataOutBase::DataOutFilterFlags flags(true, true);
  DataOutBase::DataOutFilter data_filter(flags);

  data_out.write_filtered_data(data_filter);
  data_out.write_hdf5_parallel(data_filter, output_dir + "/solution.h5", mpi_comm);

  auto new_xdmf_entry = data_out.create_xdmf_entry(
      data_filter, "solution.h5", step, mpi_comm);
  static std::vector<XDMFEntry> xdmf_entries;
  xdmf_entries.push_back(new_xdmf_entry);
  data_out.write_xdmf_file(xdmf_entries, output_dir + "/solution.xdmf", mpi_comm);
}

