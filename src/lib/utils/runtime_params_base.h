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

  prm.enter_subsection("boundary conditions");
  {
    prm.declare_entry("n_bcs", "0", Patterns::Integer());

    for (unsigned int i = 0; i < 20; ++i) 
      prm.declare_entry(
        "bc" + std::to_string(i), "(0,0,0.0,uniform)", Patterns::Anything());
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
  }
  prm.leave_subsection();

  prm.enter_subsection("finite element");
  {
    degree_vel = prm.get_integer("degree_vel");
    degree_pre = prm.get_integer("degree_pre");
  }
  prm.leave_subsection();

  prm.enter_subsection("boundary conditions");
  {
    const unsigned int n_bcs = prm.get_integer("n_bcs");
  
    for (unsigned int i = 0; i < n_bcs; ++i)
    {
      const std::string line = prm.get("bc" + std::to_string(i));
    
      std::string cleaned_line = line;
      cleaned_line.erase(std::remove(cleaned_line.begin(), cleaned_line.end(), '('), cleaned_line.end());
      cleaned_line.erase(std::remove(cleaned_line.begin(), cleaned_line.end(), ')'), cleaned_line.end());
    
      const auto tokens = Utilities::split_string_list(cleaned_line);
    
      AssertThrow(tokens.size() == 4, ExcMessage("Invalid BC format: " + line));
    
      bcs.push_back({
        static_cast<unsigned int>(std::stoi(tokens[0])),
        static_cast<unsigned int>(std::stoi(tokens[1])),
        std::stod(tokens[2]),
        tokens[3]
      });
    }
  }
  prm.leave_subsection();

  verbose = prm.get_bool("verbose");

  output_dir = prm.get("output_dir");
  output_interval = prm.get_integer("output_interval");
}
