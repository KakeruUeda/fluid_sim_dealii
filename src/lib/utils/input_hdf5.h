#pragma once
#include <hdf5.h>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include "boundary_conditions.h"

inline bool h5_exists(hid_t parent, const char* name)
{
  htri_t ex = H5Lexists(parent, name, H5P_DEFAULT);
  return ex > 0;
}

template <typename T>
std::vector<T> h5_read_array_1d(hid_t parent, const char* dset_name,
                                hid_t native_type)
{
  std::vector<T> out;

  hid_t dset = H5Dopen2(parent, dset_name, H5P_DEFAULT);
  if (dset < 0)
    throw std::runtime_error(std::string("No dataset: ") + dset_name);
  hid_t space = H5Dget_space(dset);
  if (H5Sget_simple_extent_ndims(space) != 1)
  {
    H5Sclose(space);
    H5Dclose(dset);
    throw std::runtime_error("Dataset is not array 1d: " +
                             std::string(dset_name));
  }
  hsize_t n;
  H5Sget_simple_extent_dims(space, &n, nullptr);
  out.resize(static_cast<size_t>(n));
  H5Dread(dset, native_type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
          out.data());
  H5Sclose(space);
  H5Dclose(dset);

  return out;
}

template <typename T>
T h5_read_scalar(hid_t parent, const char* dset_name,
                 hid_t native_type)
{
  hid_t dset = H5Dopen2(parent, dset_name, H5P_DEFAULT);
  if (dset < 0)
    throw std::runtime_error(std::string("No dataset: ") + dset_name);

  hid_t space = H5Dget_space(dset);
  if (space < 0)
  {
    H5Dclose(dset);
    throw std::runtime_error(std::string("H5Dget_space failed: ") +
                             dset_name);
  }

  const int rank = H5Sget_simple_extent_ndims(space);
  if (rank != 0)
  {
    H5Sclose(space);
    H5Dclose(dset);
    throw std::runtime_error("Dataset is not scalar (rank 0): " +
                             std::string(dset_name));
  }

  T value{};
  if (H5Dread(dset, native_type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              &value) < 0)
  {
    H5Sclose(space);
    H5Dclose(dset);
    throw std::runtime_error(std::string("H5Dread failed: ") +
                             dset_name);
  }

  H5Sclose(space);
  H5Dclose(dset);
  return value;
}

inline std::vector<std::string> h5_list_children(hid_t parent)
{
  std::vector<std::string> names;
  H5G_info_t info;
  H5Gget_info(parent, &info);
  names.reserve(static_cast<size_t>(info.nlinks));
  for (hsize_t i = 0; i < info.nlinks; ++i)
  {
    ssize_t len =
        H5Lget_name_by_idx(parent, ".", H5_INDEX_NAME, H5_ITER_INC, i,
                           nullptr, 0, H5P_DEFAULT);
    std::string name(static_cast<size_t>(len) + 1, '\0');
    H5Lget_name_by_idx(parent, ".", H5_INDEX_NAME, H5_ITER_INC, i,
                       name.data(), name.size(), H5P_DEFAULT);
    name.pop_back();
    names.push_back(std::move(name));
  }
  return names;
}

template <int dim>
inline void assign_center_from_vec(const std::vector<double>& cvec,
                                   std::array<double, dim>& center)
{
  if (cvec.size() == static_cast<size_t>(dim))
  {
    for (int i = 0; i < dim; ++i) center[i] = cvec[i];
  }
  else if (cvec.size() == 3 && dim == 2)
  {
    center[0] = cvec[0];
    center[1] = cvec[1];
  }
  else if (cvec.size() == 2 && dim == 3)
  {
    center[0] = cvec[0];
    center[1] = cvec[1];
    center[2] = 0.0;
  }
  else
  {
    throw std::runtime_error(
        "parabolic/center length mismatch: got " +
        std::to_string(cvec.size()) + ", expected " +
        std::to_string(dim) + " (or compatible)");
  }
}

template <int dim>
void process_bcs_from_h5(BCData<dim>& bc_data,
                         ConditionalOStream& pcout,
                         const std::string path)
{
  hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0)
  {
    pcout << "HDF5 open failed: " << path << std::endl;
    throw std::runtime_error("Failed to open HDF5");
  }

  BCData<dim> data;

  // time
  data.time = h5_read_array_1d<double>(file, "time", H5T_NATIVE_DOUBLE);
  const std::size_t n_steps = data.time.size();

  // bc groups
  auto names = h5_list_children(file);
  std::regex bc_pat(R"(bc\d+)");
  for (const auto& name : names)
  {
    if (!std::regex_match(name, bc_pat)) continue;

    hid_t g_bc = H5Gopen2(file, name.c_str(), H5P_DEFAULT);
    if (g_bc < 0) continue;

    const unsigned int bc_id = static_cast<unsigned int>(
        h5_read_scalar<int>(g_bc, "id", H5T_NATIVE_INT));

    // dir subgroups
    auto bc_children = h5_list_children(g_bc);
    std::regex dir_pat(R"(dir(\d+))");
    for (const auto& child : bc_children)
    {
      std::smatch m;
      if (!std::regex_match(child, m, dir_pat)) continue;

      const int dir_from_name = std::stoi(m[1]);
      hid_t g_dir = H5Gopen2(g_bc, child.c_str(), H5P_DEFAULT);
      if (g_dir < 0)
      {
        H5Gclose(g_bc);
        H5Fclose(file);
        throw std::runtime_error(
            "Failed to open dir group: " + child + " in " + name);
      }

      BoundaryCondition<dim> bc;
      bc.id = bc_id;

      // dir consistency
      int dir_scalar = dir_from_name;
      if (h5_exists(g_dir, "dir"))
      {
        dir_scalar =
            h5_read_scalar<int>(g_dir, "dir", H5T_NATIVE_INT);
        if (dir_scalar != dir_from_name)
        {
          H5Gclose(g_dir);
          H5Gclose(g_bc);
          H5Fclose(file);
          throw std::runtime_error(
              "dir name and dataset mismatch in " + name + "/" +
              child);
        }
      }
      bc.dir = static_cast<unsigned int>(dir_scalar);

      // value
      bc.value = h5_read_array_1d<double>(g_dir, "value", H5T_NATIVE_DOUBLE);
      if (bc.value.size() != n_steps)
      {
        H5Gclose(g_dir);
        H5Gclose(g_bc);
        H5Fclose(file);
        throw std::runtime_error("value length != time length in " +
                                 name + "/" + child);
      }

      // profile by subgroup existence
      ProfileType profile;
      if (h5_exists(g_dir, "parabolic"))
        profile = ProfileType::Parabolic;
      else if (h5_exists(g_dir, "uniform"))
        profile = ProfileType::Uniform;
      else
      {
        H5Gclose(g_dir);
        H5Gclose(g_bc);
        H5Fclose(file);
        throw std::runtime_error(
            "Unknown profile (neither 'uniform' nor 'parabolic') "
            "in " +
            name + "/" + child);
      }
      bc.profile = profile;

      if (profile == ProfileType::Parabolic)
      {
        hid_t pg = H5Gopen2(g_dir, "parabolic", H5P_DEFAULT);
        if (pg < 0)
        {
          H5Gclose(g_dir);
          H5Gclose(g_bc);
          H5Fclose(file);
          throw std::runtime_error("parabolic group open failed in " +
                                   name + "/" + child);
        }

        ParabolicProfile<dim> pp;
        pp.radius =
            h5_read_scalar<double>(pg, "radius", H5T_NATIVE_DOUBLE);

        std::vector<double> cvec;
        cvec = h5_read_array_1d<double>(pg, "center", H5T_NATIVE_DOUBLE);
        assign_center_from_vec<dim>(cvec, pp.center);

        bc.parabolic = pp;
        H5Gclose(pg);
      }

      data.bcs.push_back(std::move(bc));
      H5Gclose(g_dir);
    }

    H5Gclose(g_bc);
  }

  H5Fclose(file);

  bc_data = std::move(data);
}
