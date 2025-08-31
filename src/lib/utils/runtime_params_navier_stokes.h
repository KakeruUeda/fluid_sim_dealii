#pragma once
#include "runtime_params_base.h"

class RuntimeParams_NavierStokes : public RuntimeParams
{
public:
    void declare_params() override;
    void read_params(const std::string &filename) override;

    double dt;
    double t_end;

    double U;
    double L;
    double mu;
    double rho;
    double Re;

    std::string bc_data_path;
};

void RuntimeParams_NavierStokes::declare_params()
{
  RuntimeParams::declare_params();

  prm.enter_subsection("fluid");
  {
    prm.declare_entry("dt", "1.0e-3", Patterns::Double(0.), "Time step size");
    prm.declare_entry("t_end", "1.0", Patterns::Double(0.), "End time of simulation");
    prm.declare_entry("mu", "1.0", Patterns::Double(0.), "Fluid viscosity mu");
    prm.declare_entry("rho", "1.0", Patterns::Double(0.), "Fluid density rho");
  }
  prm.leave_subsection();

  prm.enter_subsection("boundary conditions");
  {
    prm.declare_entry("bc_data_path", "",
                      Patterns::FileName(), "Path to bc file");
  }
  prm.leave_subsection();
}

void RuntimeParams_NavierStokes::read_params(
    const std::string &filename)
{
  RuntimeParams::read_params(filename);

  prm.enter_subsection("fluid");
  {
    dt = prm.get_double("dt");
    t_end = prm.get_double("t_end");
    mu = prm.get_double("mu");
    rho = prm.get_double("rho");
  }
  prm.leave_subsection();

  prm.enter_subsection("boundary conditions");
  {
    bc_data_path = prm.get("bc_data_path");
  }
  prm.leave_subsection();
}

