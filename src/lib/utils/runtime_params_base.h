#pragma once
#include <deal.II/base/parameter_handler.h>

class RuntimeParams
{
 public:
  RuntimeParams() = default;
  virtual ~RuntimeParams() = default;

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

  std::string mesh_dir;

 protected:
  ParameterHandler prm;
};

void RuntimeParams::declare_params()
{
  prm.enter_subsection("mesh");
  {
    prm.declare_entry("mesh_dir", "../../mesh/aneurysm.msh",
                      Patterns::FileName(), "Path to mesh file");
    prm.declare_entry("inlet_label", "4", Patterns::Integer(),
                      "Boundary ID for inlet");
    prm.declare_entry("outlet_label", "6", Patterns::Integer(),
                      "Boundary ID for outlet");
    prm.declare_entry("wall_label", "5", Patterns::Integer(),
                      "Boundary ID for wall");
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
    mesh_dir = prm.get("mesh_dir");
    inlet_label = prm.get_integer("inlet_label");
    outlet_label = prm.get_integer("outlet_label");
    wall_label = prm.get_integer("wall_label");
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
