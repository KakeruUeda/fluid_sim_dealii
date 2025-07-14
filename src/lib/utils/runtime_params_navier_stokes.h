#pragma once
#include "runtime_params_base.h"

class RuntimeParams_NavierStokes : public RuntimeParams
{
public:
    void read_params(const std::string &filename) override;

    double dt;
    double t_end;

    double U;
    double L;
    double mu;
    double rho;
    
    double Re;
};

void RuntimeParams_NavierStokes::read_params(
    const std::string &filename)
{
  RuntimeParams::read_params(filename);

  prm.enter_subsection("fluid");
  {
    dt = prm.get_integer("dt");
    t_end = prm.get_integer("t_end");
    mu = prm.get_double("mu");
    rho = prm.get_double("rho");
  }
  prm.leave_subsection();
}

