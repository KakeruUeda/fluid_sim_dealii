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

static std::vector<XDMFEntry> xdmf_entries;

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
  data_out.write_hdf5_parallel(
    data_filter, output_dir+"/solution_"+std::to_string(step)+".h5", mpi_comm);

  auto new_entry = data_out.create_xdmf_entry(
    data_filter, "solution_"+std::to_string(step)+".h5",
    static_cast<double>(step), mpi_comm);

  xdmf_entries.push_back(new_entry);

  // data_out.write_xdmf_file(xdmf_entries, output_dir + "/solution.xdmf", mpi_comm);
}

template <int dim>
void write_final_xdmf(const std::string &output_dir, MPI_Comm &mpi_comm)
{
  DataOut<dim> data_out;
  data_out.write_xdmf_file(xdmf_entries, output_dir + "/solution.xdmf", mpi_comm);
}

template <int dim>
void output_results_pvd(const std::string &output_dir,
                        const Triangulation<dim> &triangulation,
                        DoFHandler<dim> &dof_handler,
                        PETScWrappers::MPI::Vector &solution,
                        MPI_Comm mpi_comm,
                        const unsigned int this_mpi_proc,
                        const unsigned int step)
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  Vector<double> local_solution;
  local_solution = solution;

  std::vector<std::string> solution_names(dim, "velocity");
  solution_names.emplace_back("pressure");
  data_out.add_data_vector(local_solution, solution_names);

  std::vector<unsigned int> partition_int(triangulation.n_active_cells());
  GridTools::get_subdomain_association(triangulation, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  DataOutBase::VtkFlags vtk_flags;
  vtk_flags.compression_level = DataOutBase::CompressionLevel::best_speed;
  data_out.set_flags(vtk_flags);

  const std::string pvtu_filename = data_out.write_vtu_with_pvtu_record(
      output_dir + "/", "solution", step, mpi_comm, 4);

  if (this_mpi_proc == 0)
  {
    static std::vector<std::pair<double, std::string>> times_and_names;
    times_and_names.emplace_back(step, pvtu_filename);
    std::ofstream pvd_output(output_dir + "/solution.pvd");
    DataOutBase::write_pvd_record(pvd_output, times_and_names);
  }
}