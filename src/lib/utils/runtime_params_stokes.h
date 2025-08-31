#pragma once
#include "runtime_params_base.h"

class RuntimeParams_Stokes : public RuntimeParams
{
public:
    void declare_params() override;
    void read_params(const std::string &filename) override;

    double mu;
    std::string bc_data_path;
};

void RuntimeParams_Stokes::declare_params()
{
  RuntimeParams::declare_params();
  
  prm.enter_subsection("fluid");
  {
    prm.declare_entry("mu", "1.0", Patterns::Double(0.), "Fluid viscosity mu");
  }
  prm.leave_subsection();

  prm.enter_subsection("boundary conditions");
  {
    prm.declare_entry("bc_data_path", "",
                      Patterns::FileName(), "Path to bc file");
  }
  prm.leave_subsection();
}

void RuntimeParams_Stokes::read_params(
    const std::string &filename)
{
  declare_params();
  RuntimeParams::read_params(filename);
  
  prm.enter_subsection("fluid");
  mu = prm.get_double("mu");
    
  prm.leave_subsection();

  prm.enter_subsection("boundary conditions");
  {
    bc_data_path = prm.get("bc_data_path");
  }
  prm.leave_subsection();
}