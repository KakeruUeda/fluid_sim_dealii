#pragma once
#include <deal.II/base/parameter_handler.h>
#include <boundary_conditions.h>

class RuntimeParams
{
 public:
  RuntimeParams() = default;
  virtual ~RuntimeParams() = default;

  std::vector<BoundaryConditions> bcs;

  virtual void declare_params();
  virtual void read_params(const std::string& filename);

  unsigned int degree_vel;
  unsigned int degree_pre;

  unsigned int inlet_label;
  unsigned int outlet_label;
  unsigned int wall_label;

  std::string output_dir;

  bool verbose;
  unsigned int output_interval;

  std::string mesh_path;

 protected:
  ParameterHandler prm;
};

void RuntimeParams::declare_params()
{
  prm.enter_subsection("mesh");
  {
    prm.declare_entry("mesh_path", "",
                      Patterns::FileName(), "Path to mesh file");
  }
  prm.leave_subsection();

  prm.enter_subsection("finite element");
  {
    prm.declare_entry("degree_vel", "1", Patterns::Integer(1),
                      "Degree of velocity FE");
    prm.declare_entry("degree_pre", "1", Patterns::Integer(1),
                      "Degree of pressure FE");
  }
  prm.leave_subsection();

  prm.declare_entry("verbose", "true", Patterns::Bool(),
                    "Verbose output");
  prm.declare_entry("output_dir", "tmp", Patterns::FileName(),
                    "Output dir");
  prm.declare_entry("output_interval", "10", Patterns::Integer(10),
                    "Output interval");
}

void RuntimeParams::read_params(const std::string& filename)
{
  declare_params();

  std::ifstream file(filename);
  AssertThrow(file, ExcFileNotOpen(filename));

  prm.parse_input(file);

  prm.enter_subsection("mesh");
  {
    mesh_path = prm.get("mesh_path");
  }
  prm.leave_subsection();

  prm.enter_subsection("finite element");
  {
    degree_vel = prm.get_integer("degree_vel");
    degree_pre = prm.get_integer("degree_pre");
  }
  prm.leave_subsection();

  verbose = prm.get_bool("verbose");

  output_dir = prm.get("output_dir");
  output_interval = prm.get_integer("output_interval");
}
