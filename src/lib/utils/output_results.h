#pragma once
#include <hdf5.h>
#include <custom_io_dealii.h>

static std::vector<XDMFEntry> xdmf_entries;
static std::vector<CustomXDMFEntry> custom_xdmf_entries;

template <int dim>
void output_results(std::string output_dir,
                    const Triangulation<dim>& triangulation, 
                    DoFHandler<dim> &dof_handler, 
                    Vector<double> &solution,
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
void write_group_hdf5(const dealii::DataOutBase::DataOutFilter &filter,
                      const std::string &filename,
                      const std::string &group_name,
                      MPI_Comm /*mpi_comm*/) // not used for now
{
  hid_t file_id;

  {
    std::ifstream f(filename.c_str());
    if (f.good())
      file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    else
      file_id = H5Fcreate(filename.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
  }
  AssertThrow(file_id >= 0, dealii::ExcIO());

  if (group_name == "/step00000000")
  {
    std::vector<double> node_data;
    filter.fill_node_data(node_data);
    const hsize_t n_nodes = filter.n_nodes();
    const hsize_t spacedim = node_data.size() / n_nodes;
    hsize_t node_dims[2] = {n_nodes, spacedim};
    hid_t node_space = H5Screate_simple(2, node_dims, nullptr);
    hid_t node_ds = H5Dcreate(file_id, "/nodes",
                              H5T_NATIVE_DOUBLE, node_space,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(node_ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, node_data.data());
    H5Sclose(node_space);
    H5Dclose(node_ds);

    std::vector<unsigned int> cell_data;
    filter.fill_cell_data(0, cell_data);
    const hsize_t n_cells = filter.n_cells();
    const hsize_t vertices_per_cell = cell_data.size() / n_cells;
    hsize_t cell_dims[2] = {n_cells, vertices_per_cell};
    hid_t cell_space = H5Screate_simple(2, cell_dims, nullptr);
    hid_t cell_ds = H5Dcreate(file_id, "/cells",
                              H5T_NATIVE_UINT, cell_space,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(cell_ds, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, cell_data.data());
    H5Sclose(cell_space);
    H5Dclose(cell_ds);
  }

  hid_t group_id = H5Gcreate(file_id, group_name.c_str(),
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  AssertThrow(group_id >= 0, dealii::ExcIO());

  const hsize_t n_nodes = filter.n_nodes();
  for (unsigned int i = 0; i < filter.n_data_sets(); ++i)
  {
    const std::string &name = filter.get_data_set_name(i);
    const unsigned int d = filter.get_data_set_dim(i);
    const double *data_ptr = filter.get_data_set(i);

    hsize_t dims[2] = {n_nodes, d};
    hid_t data_space = H5Screate_simple(2, dims, nullptr);
    hid_t data_ds = H5Dcreate(group_id, name.c_str(),
                              H5T_NATIVE_DOUBLE, data_space,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(data_ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_ptr);
    H5Sclose(data_space);
    H5Dclose(data_ds);
  }

  H5Gclose(group_id);
  H5Fclose(file_id);
}

template <int dim>
void write_group_hdf5_parallel(const dealii::DataOutBase::DataOutFilter &filter,
                      const std::string &filename,
                      const std::string &group_name,
                      MPI_Comm mpi_comm)
{
  herr_t status;
  hid_t file_id, plist_id, dxpl_id;

  const std::uint64_t local_nodes = filter.n_nodes();
  const std::uint64_t local_cells = filter.n_cells();

  std::uint64_t global_nodes = 0, global_cells = 0;
  std::uint64_t offset_nodes = 0, offset_cells = 0;

  MPI_Allreduce(&local_nodes, &global_nodes, 1,
                MPI_UINT64_T, MPI_SUM, mpi_comm);
  MPI_Allreduce(&local_cells, &global_cells, 1,
                MPI_UINT64_T, MPI_SUM, mpi_comm);

  MPI_Exscan(&local_nodes, &offset_nodes, 1, MPI_UINT64_T, MPI_SUM, mpi_comm);
  MPI_Exscan(&local_cells, &offset_cells, 1, MPI_UINT64_T, MPI_SUM, mpi_comm);

  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  status = H5Pset_fapl_mpio(plist_id, mpi_comm, MPI_INFO_NULL);
  AssertThrow(status >= 0, dealii::ExcIO());

  if (group_name == "/step00000000")
    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  else 
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, plist_id);
  // {
  //   std::ifstream f(filename.c_str());
  //   if (f.good())
  //     file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, plist_id);
  //   else
  //     file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  // }
  AssertThrow(file_id >= 0, dealii::ExcIO());
  H5Pclose(plist_id);

  dxpl_id = H5Pcreate(H5P_DATASET_XFER);
  status = H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE);
  AssertThrow(status >= 0, dealii::ExcIO());

  if (group_name == "/step00000000")
  {
    std::vector<double> node_data;
    filter.fill_node_data(node_data);
    const hsize_t spacedim = node_data.size() / local_nodes;
    hsize_t full_dims[2] = {global_nodes, spacedim};
    hsize_t count[2]     = {local_nodes, spacedim};
    hsize_t offset[2]    = {offset_nodes, 0};

    hid_t filespace = H5Screate_simple(2, full_dims, nullptr);
    hid_t memspace  = H5Screate_simple(2, count, nullptr);
    hid_t dset      = H5Dcreate(file_id, "/nodes", H5T_NATIVE_DOUBLE,
                                filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, nullptr, count, nullptr);
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace, dxpl_id, node_data.data());
    H5Sclose(memspace); H5Sclose(filespace); H5Dclose(dset);

    std::vector<unsigned int> cell_data;
    filter.fill_cell_data(offset_nodes, cell_data);
    const hsize_t vertices_per_cell = cell_data.size() / local_cells;
    hsize_t full_cell_dims[2] = {global_cells, vertices_per_cell};
    hsize_t cell_count[2]     = {local_cells, vertices_per_cell};
    hsize_t cell_offset[2]    = {offset_cells, 0};

    hid_t cellspace = H5Screate_simple(2, full_cell_dims, nullptr);
    hid_t cellmem   = H5Screate_simple(2, cell_count, nullptr);
    hid_t cell_dset = H5Dcreate(file_id, "/cells", H5T_NATIVE_UINT,
                                cellspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sselect_hyperslab(cellspace, H5S_SELECT_SET, cell_offset, nullptr, cell_count, nullptr);
    H5Dwrite(cell_dset, H5T_NATIVE_UINT, cellmem, cellspace, dxpl_id, cell_data.data());
    H5Sclose(cellmem); H5Sclose(cellspace); H5Dclose(cell_dset);
  }

  hid_t group_id = H5Gcreate(file_id, group_name.c_str(),
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  AssertThrow(group_id >= 0, dealii::ExcIO());

  for (unsigned int i = 0; i < filter.n_data_sets(); ++i)
  {
    const std::string &name = filter.get_data_set_name(i);
    const unsigned int d = filter.get_data_set_dim(i);
    const double *data_ptr = filter.get_data_set(i);

    hsize_t full_dims[2] = {global_nodes, d};
    hsize_t count[2]     = {local_nodes, d};
    hsize_t offset[2]    = {offset_nodes, 0};

    hid_t filespace = H5Screate_simple(2, full_dims, nullptr);
    hid_t memspace  = H5Screate_simple(2, count, nullptr);
    hid_t dset      = H5Dcreate(group_id, name.c_str(), H5T_NATIVE_DOUBLE,
                                filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, nullptr, count, nullptr);
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace, dxpl_id, data_ptr);

    H5Sclose(memspace); H5Sclose(filespace); H5Dclose(dset);
  }

  H5Gclose(group_id);
  H5Pclose(dxpl_id);
  H5Fclose(file_id);
}


template <int dim>
void output_results_to_group_hdf5(std::string output_dir,
                            const Triangulation<dim>& triangulation, 
                            DoFHandler<dim> &dof_handler, 
                            Vector<double> &solution,
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

  // std::vector<unsigned int> partition_int(triangulation.n_active_cells());
  // GridTools::get_subdomain_association(triangulation, partition_int);
  // const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  // data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  DataOutBase::DataOutFilterFlags flags(true, true);
  DataOutBase::DataOutFilter data_filter(flags);
  data_out.write_filtered_data(data_filter);

  std::string group_name = "/step" + Utilities::int_to_string(step, 8);

  std::string h5_filename = output_dir + "/solution.h5";
  write_group_hdf5_parallel<dim>(data_filter, h5_filename, group_name, mpi_comm);

  auto entry = create_custom_xdmf_entry(data_out, data_filter, 
    "solution.h5", "solution.h5", step, mpi_comm);

  custom_xdmf_entries.push_back(entry);
  write_custom_xdmf_file(custom_xdmf_entries, output_dir + "/solution.xdmf", mpi_comm);
}


template <int dim>
void write_custom_xdmf_all(const std::string &output_dir, MPI_Comm &mpi_comm)
{
  write_custom_xdmf_file(custom_xdmf_entries, output_dir + "/solution.xdmf", mpi_comm);
}

template <int dim>
void write_xdmf_all(const std::string &output_dir, MPI_Comm &mpi_comm)
{
  DataOut<dim> data_out;
  data_out.write_xdmf_file(xdmf_entries, output_dir + "/solution.xdmf", mpi_comm);
}

template <int dim>
void output_results_pvd(const std::string &output_dir,
                        const Triangulation<dim> &triangulation,
                        DoFHandler<dim> &dof_handler,
                        Vector<double> &solution,
                        MPI_Comm mpi_comm,
                        const unsigned int this_mpi_proc,
                        const unsigned int step)
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  std::vector<std::string> solution_names(dim, "velocity");
  solution_names.emplace_back("pressure");
  data_out.add_data_vector(solution, solution_names);

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