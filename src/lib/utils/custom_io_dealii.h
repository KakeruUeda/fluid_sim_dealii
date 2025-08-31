#pragma once

#include <deal.II/base/config.h>

#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi_stub.h>
#include <deal.II/base/point.h>
#include <deal.II/base/table.h>

#include <deal.II/grid/reference_cell.h>

#include <deal.II/numerics/data_component_interpretation.h>

// To be able to serialize XDMFEntry
#include <boost/serialization/map.hpp>

#include <limits>
#include <ostream>
#include <string>
#include <tuple>
#include <typeinfo>
#include <vector>

class CustomXDMFEntry
{
 public:
  CustomXDMFEntry();
  CustomXDMFEntry(const std::string& filename, const double time,
                  const std::uint64_t nodes,
                  const std::uint64_t cells, const unsigned int dim,
                  const ReferenceCell& cell_type);
  CustomXDMFEntry(const std::string& mesh_filename,
                  const std::string& solution_filename,
                  const double time, const std::uint64_t nodes,
                  const std::uint64_t cells, const unsigned int dim,
                  const ReferenceCell& cell_type);
  CustomXDMFEntry(const std::string& mesh_filename,
                  const std::string& solution_filename,
                  const double time, const std::uint64_t nodes,
                  const std::uint64_t cells, const unsigned int dim,
                  const unsigned int spacedim,
                  const ReferenceCell& cell_type);
  void add_attribute(const std::string& attr_name,
                     const std::string& attr_h5_path,
                     const unsigned int dimension);
  template <class Archive>
  void serialize(Archive& ar, const unsigned int /*version*/)
  {
    ar & valid & h5_sol_filename & h5_mesh_filename & entry_time &
        num_nodes & num_cells & dimension & space_dimension &
        cell_type & attributes;
  }
  std::string get_xdmf_content(const unsigned int indent_level) const;

 private:
  bool valid;
  std::string h5_sol_filename;
  std::string h5_mesh_filename;
  double entry_time;
  std::uint64_t num_nodes;
  std::uint64_t num_cells;
  unsigned int dimension;
  unsigned int space_dimension;
  ReferenceCell cell_type;
  std::map<std::string, std::pair<std::string, unsigned int>> attributes;
};

CustomXDMFEntry::CustomXDMFEntry()
    : valid(false),
      h5_sol_filename(""),
      h5_mesh_filename(""),
      entry_time(0.0),
      num_nodes(numbers::invalid_unsigned_int),
      num_cells(numbers::invalid_unsigned_int),
      dimension(numbers::invalid_unsigned_int),
      space_dimension(numbers::invalid_unsigned_int),
      cell_type()
{
}

CustomXDMFEntry::CustomXDMFEntry(const std::string& filename,
                                 const double time,
                                 const std::uint64_t nodes,
                                 const std::uint64_t cells,
                                 const unsigned int dim,
                                 const ReferenceCell& cell_type)
    : CustomXDMFEntry(filename, filename, time, nodes, cells, dim,
                      dim, cell_type)
{
}

CustomXDMFEntry::CustomXDMFEntry(const std::string& mesh_filename,
                                 const std::string& solution_filename,
                                 const double time,
                                 const std::uint64_t nodes,
                                 const std::uint64_t cells,
                                 const unsigned int dim,
                                 const ReferenceCell& cell_type)
    : CustomXDMFEntry(mesh_filename, solution_filename, time, nodes,
                      cells, dim, dim, cell_type)
{
}

namespace
{
ReferenceCell cell_type_hex_if_invalid(const ReferenceCell& cell_type,
                                       const unsigned int dimension)
{
  if (cell_type == ReferenceCells::Invalid)
  {
    switch (dimension)
    {
      case 0:
        return ReferenceCells::get_hypercube<0>();
      case 1:
        return ReferenceCells::get_hypercube<1>();
      case 2:
        return ReferenceCells::get_hypercube<2>();
      case 3:
        return ReferenceCells::get_hypercube<3>();
      default:
        AssertThrow(false, ExcMessage("Invalid dimension"));
    }
  }
  else
    return cell_type;
}
}  // namespace

CustomXDMFEntry::CustomXDMFEntry(const std::string& mesh_filename,
                                 const std::string& solution_filename,
                                 const double time,
                                 const std::uint64_t nodes,
                                 const std::uint64_t cells,
                                 const unsigned int dim,
                                 const unsigned int spacedim,
                                 const ReferenceCell& cell_type_)
    : valid(true),
      h5_sol_filename(solution_filename),
      h5_mesh_filename(mesh_filename),
      entry_time(time),
      num_nodes(nodes),
      num_cells(cells),
      dimension(dim),
      space_dimension(spacedim),
      cell_type(cell_type_hex_if_invalid(cell_type_, dim))
{
}

void CustomXDMFEntry::add_attribute(const std::string& attr_name,
                                    const std::string& attr_h5_path,
                                    const unsigned int dimension)
{
  attributes[attr_name] = std::make_pair(attr_h5_path, dimension);
}

namespace
{
std::string indent(const unsigned int indent_level)
{
  std::string res = "";
  for (unsigned int i = 0; i < indent_level; ++i) res += "  ";
  return res;
}
}  // namespace

std::string CustomXDMFEntry::get_xdmf_content(
    const unsigned int indent_level) const
{
  if (!valid) return "";

  std::stringstream ss;
  ss.precision(12);
  ss << indent(indent_level + 0)
     << "<Grid Name=\"mesh\" GridType=\"Uniform\">\n";
  ss << indent(indent_level + 1) << "<Time Value=\"" << entry_time
     << "\"/>\n";
  ss << indent(indent_level + 1) << "<Geometry GeometryType=\""
     << (space_dimension <= 2 ? "XY" : "XYZ") << "\">\n";
  ss << indent(indent_level + 2) << "<DataItem Dimensions=\""
     << num_nodes << " "
     << (space_dimension <= 2 ? 2 : space_dimension)
     << "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n";
  ss << indent(indent_level + 3) << h5_mesh_filename << ":/nodes\n";
  ss << indent(indent_level + 2) << "</DataItem>\n";
  ss << indent(indent_level + 1) << "</Geometry>\n";

  // If we have cells defined, use the topology corresponding to the
  // dimension
  if (num_cells > 0)
  {
    ss << indent(indent_level + 1) << "<Topology TopologyType=\"";

    if (dimension == 0)
    {
      ss << "Polyvertex";
    }
    else if (dimension == 1)
    {
      ss << "Polyline";
    }
    else if (dimension == 2)
    {
      Assert(cell_type == ReferenceCells::Quadrilateral ||
                 cell_type == ReferenceCells::Triangle,
             ExcNotImplemented());

      if (cell_type == ReferenceCells::Quadrilateral)
      {
        ss << "Quadrilateral";
      }
      else  // if (cell_type == ReferenceCells::Triangle)
      {
        ss << "Triangle";
      }
    }
    else if (dimension == 3)
    {
      Assert(cell_type == ReferenceCells::Hexahedron ||
                 cell_type == ReferenceCells::Tetrahedron,
             ExcNotImplemented());

      if (cell_type == ReferenceCells::Hexahedron)
      {
        ss << "Hexahedron";
      }
      else  // if (reference_cell == ReferenceCells::Tetrahedron)
      {
        ss << "Tetrahedron";
      }
    }

    ss << "\" NumberOfElements=\"" << num_cells;
    if (dimension == 0)
      ss << "\" NodesPerElement=\"1\">\n";
    else if (dimension == 1)
      ss << "\" NodesPerElement=\"2\">\n";
    else
      // no "NodesPerElement" for dimension 2 and higher
      ss << "\">\n";

    ss << indent(indent_level + 2) << "<DataItem Dimensions=\""
       << num_cells << " " << cell_type.n_vertices()
       << "\" NumberType=\"UInt\" Format=\"HDF\">\n";

    ss << indent(indent_level + 3) << h5_mesh_filename << ":/cells\n";
    ss << indent(indent_level + 2) << "</DataItem>\n";
    ss << indent(indent_level + 1) << "</Topology>\n";
  }
  // Otherwise, we assume the points are isolated in space and use a
  // Polyvertex topology
  else
  {
    ss << indent(indent_level + 1)
       << "<Topology TopologyType=\"Polyvertex\" NumberOfElements=\""
       << num_nodes << "\">\n";
    ss << indent(indent_level + 1) << "</Topology>\n";
  }

  for (const auto& attr : attributes)
  {
    const std::string& name = attr.first;
    const std::string& path = attr.second.first;
    unsigned int dim = attr.second.second;

    ss << indent(indent_level + 1) << "<Attribute Name=\"" << name
       << "\" AttributeType=\"" << (dim > 1 ? "Vector" : "Scalar")
       << "\" Center=\"Node\">\n";

    ss << indent(indent_level + 2) << "<DataItem Dimensions=\""
       << num_nodes << " " << (dim > 1 ? 3 : 1)
       << "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n";

    ss << indent(indent_level + 3) << h5_sol_filename << ":" << path
       << '\n';
    ss << indent(indent_level + 2) << "</DataItem>\n";
    ss << indent(indent_level + 1) << "</Attribute>\n";
  }

  ss << indent(indent_level + 0) << "</Grid>\n";

  return ss.str();
}

void write_custom_xdmf_file(
    const std::vector<CustomXDMFEntry>& entries,
    const std::string& filename, const MPI_Comm comm)
{
#ifdef DEAL_II_WITH_MPI
  const int myrank = Utilities::MPI::this_mpi_process(comm);
#else
  (void)comm;
  const int myrank = 0;
#endif

  // Only rank 0 process writes the XDMF file
  if (myrank == 0)
  {
    std::ofstream xdmf_file(filename);

    xdmf_file << "<?xml version=\"1.0\" ?>\n";
    xdmf_file << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    xdmf_file << "<Xdmf Version=\"2.0\">\n";
    xdmf_file << "  <Domain>\n";
    xdmf_file
        << "    <Grid Name=\"CellTime\" GridType=\"Collection\" "
           "CollectionType=\"Temporal\">\n";

    for (const auto& entry : entries)
    {
      xdmf_file << entry.get_xdmf_content(3);
    }

    xdmf_file << "    </Grid>\n";
    xdmf_file << "  </Domain>\n";
    xdmf_file << "</Xdmf>\n";

    xdmf_file.close();
  }
}

template <int dim>
CustomXDMFEntry create_custom_xdmf_entry(
    const DataOut<dim>& data_out,
    const DataOutBase::DataOutFilter& data_filter,
    const std::string& h5_mesh_filename,
    const std::string& h5_solution_filename, const unsigned int step,
    const MPI_Comm comm)
{
  AssertThrow(
      dim == 2 || dim == 3,
      ExcMessage("XDMF only supports 2 or 3 space dimensions."));

#ifndef DEAL_II_WITH_HDF5
  // throw an exception, but first make sure the compiler does not
  // warn about the now unused function arguments
  (void)data_filter;
  (void)h5_mesh_filename;
  (void)h5_solution_filename;
  (void)cur_time;
  (void)comm;
  AssertThrow(
      false,
      ExcMessage("XDMF support requires HDF5 to be turned on."));

  return {};

#else

  std::uint64_t local_node_cell_count[2], global_node_cell_count[2];

  local_node_cell_count[0] = data_filter.n_nodes();
  local_node_cell_count[1] = data_filter.n_cells();

  const int myrank = Utilities::MPI::this_mpi_process(comm);
  // And compute the global total
  int ierr = MPI_Allreduce(
      local_node_cell_count, global_node_cell_count, 2,
      Utilities::MPI::mpi_type_id_for_type<std::uint64_t>, MPI_SUM,
      comm);
  AssertThrowMPI(ierr);

  // The implementation is a bit complicated because we are supposed
  // to return the correct data on rank 0 and an empty object on all
  // other ranks but all information (for example the attributes) are
  // only available on ranks that have any cells. We will identify the
  // smallest rank that has data and then communicate from this rank
  // to rank 0 (if they are different ranks).

  const bool have_data = (data_filter.n_nodes() > 0);
  MPI_Comm split_comm;
  {
    const int key = myrank;
    const int color = (have_data ? 1 : 0);
    const int ierr = MPI_Comm_split(comm, color, key, &split_comm);
    AssertThrowMPI(ierr);
  }

  const bool am_i_first_rank_with_data =
      have_data &&
      (Utilities::MPI::this_mpi_process(split_comm) == 0);

  ierr = MPI_Comm_free(&split_comm);
  AssertThrowMPI(ierr);

  const int tag = 47381;

  // Output the XDMF file only on the root process of all ranks with
  // data:
  if (am_i_first_rank_with_data)
  {
    const auto& patches = data_out.get_patches();
    Assert(patches.size() > 0, DataOutBase::ExcNoPatches());

    // // We currently don't support writing mixed meshes:
    // if constexpr (running_in_debug_mode())
    //   {
    //     for (const auto &patch : patches)
    //       Assert(patch.reference_cell == patches[0].reference_cell,
    //              ExcNotImplemented());
    //   }
    auto entry = CustomXDMFEntry(
        h5_mesh_filename, h5_solution_filename, step,
        global_node_cell_count[0], global_node_cell_count[1], dim,
        patches[0].reference_cell);

    const unsigned int n_data_sets = data_filter.n_data_sets();

    std::string group = "step" + Utilities::int_to_string(step, 8);
    for (unsigned int i = 0; i < n_data_sets; ++i)
    {
      entry.add_attribute(
          data_filter.get_data_set_name(i),
          group + "/" + data_filter.get_data_set_name(i),
          data_filter.get_data_set_dim(i));
    }

    if (myrank != 0)
    {
      // send to rank 0
      const std::vector<char> buffer = Utilities::pack(entry, false);
      ierr = MPI_Send(buffer.data(), buffer.size(), MPI_BYTE, 0, tag,
                      comm);
      AssertThrowMPI(ierr);

      return {};
    }

    return entry;
  }

  if (myrank == 0 && !am_i_first_rank_with_data)
  {
    // receive the XDMF data on rank 0 if we don't have it...

    MPI_Status status;
    int ierr = MPI_Probe(MPI_ANY_SOURCE, tag, comm, &status);
    AssertThrowMPI(ierr);

    int len;
    ierr = MPI_Get_count(&status, MPI_BYTE, &len);
    AssertThrowMPI(ierr);

    std::vector<char> buffer(len);
    ierr = MPI_Recv(buffer.data(), len, MPI_BYTE, status.MPI_SOURCE,
                    tag, comm, MPI_STATUS_IGNORE);
    AssertThrowMPI(ierr);

    return Utilities::unpack<CustomXDMFEntry>(buffer, false);
  }

  // default case for any other rank is to return an empty object
  return {};
#endif
}
