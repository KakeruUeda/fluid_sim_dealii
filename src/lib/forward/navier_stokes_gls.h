/* ---------------------
 * navier_stokes.h
 *
 * Copyright (C) 2025
 * All rights reserved.
 *
 * ---------------------
 *
 * Author: Kakeru Ueda
 */

#pragma once 
#include <deal.II/base/parameter_handler.h>

#include "pde_base.h"
#include "boundary_conditions.h"
#include "runtime_params_navier_stokes.h"

#include <string>
#include <sys/stat.h>

template <int dim>
class NavierStokesGLS : public PDEBase<dim>
{
public:
  using PDEBase<dim>::mpi_comm;
  using PDEBase<dim>::pcout;
  using PDEBase<dim>::fe;
  using PDEBase<dim>::triangulation;
  using PDEBase<dim>::dof_handler;
  using PDEBase<dim>::make_grid;
  using PDEBase<dim>::bcs;

  NavierStokesGLS(
    const RuntimeParams_NavierStokes &params);
  ~NavierStokesGLS() override = default;

  void run();

private:
  const double dt;
  const double t_end;
  const double mu;
  const double rho;
  const double nu;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  const FEValuesExtractors::Vector vel_ext;
  const FEValuesExtractors::Scalar pre_ext;

  const QGaussSimplex <dim> q_gauss;

  PETScWrappers::MPI::SparseMatrix system_matrix;
  PETScWrappers::MPI::Vector solution;
  PETScWrappers::MPI::Vector solution_prev;
  PETScWrappers::MPI::Vector system_rhs;

  std::vector<Tensor<1, dim>> u_q;
  std::vector<Tensor<1, dim>> u_q_prev;
  std::vector<Tensor<2, dim>> u_q_grad;
  std::vector<Tensor<1, dim>> u_q_adv;
  std::vector<double> p_q;

  void setup_system();
  void assemble_system(const bool initial);
  void initialize();
  void solve_system();
  
  double comp_cell_length(
    FEValues<dim> &fe_values, Tensor<1, dim> &u, 
    const unsigned int dofs_per_cell, const unsigned int q);

  std::array<double, 2> comp_stab_params_1(
    const Tensor<1, dim> &u, const double h);

  std::array<double, 2> comp_stab_params_2(
    const Tensor<1, dim> &u, const double h);

  void assemble_matrix_initial(
    FullMatrix<double> &cell_matrix, FEValues<dim> &fe_values, 
    const double mu, std::array<double, 2> &tau, const unsigned int q);

  void assemble_matrix(
    FullMatrix<double> &cell_matrix, FEValues<dim> &fe_values, 
    std::array<double, 2> &tau, const unsigned int q);

  void assemble_rhs(
    Vector<double> &cell_rhs, FEValues<dim> &fe_values, 
    std::array<double, 2> &tau, const unsigned int q);

  void apply_dirichlet_boundary_conditions(
    std::map<types::global_dof_index, double>& boundary_values);
};

template <int dim>
NavierStokesGLS<dim>::NavierStokesGLS(
    const RuntimeParams_NavierStokes &params)
    : PDEBase<dim>(params)
    , dt(params.dt)
    , t_end(params.t_end)
    , mu(params.mu)
    , rho(params.rho)
    , nu(params.mu/params.rho)
    , vel_ext(0)
    , pre_ext(dim)
    , q_gauss(2)
{
  u_q.resize(q_gauss.size());
  u_q_prev.resize(q_gauss.size());
  u_q_adv.resize(q_gauss.size());
  u_q_grad.resize(q_gauss.size());
  p_q.resize(q_gauss.size());

  std::string output_dir_tmp = "outputs";
  mkdir(output_dir_tmp.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);

  this->output_dir = "outputs/" + this->output_dir;
  mkdir(this->output_dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
}

template <int dim>
void NavierStokesGLS<dim>::initialize()
{
  for (unsigned int i=0; i<q_gauss.size(); ++i)
  {
    u_q[i] = 0e0;
    u_q_prev[i] = 0e0;
    u_q_adv[i] = 0e0;
    u_q_grad[i] = 0e0;
  }
}

template <int dim>
void NavierStokesGLS<dim>::run()
{
  make_grid();
  setup_system();
  initialize();

  // assemble_system(true);
  // solve_system();

  unsigned int step_max 
  = static_cast<unsigned int>(t_end/dt) + 1;
  
  double t = 0e0;
  for (unsigned int n = 0; n < step_max; ++n)
  {
    pcout << std::left
      << " time = " << std::setw(8) << t << " [s], "
      << " step/step_max = " << std::setw(4) 
      << n << " / " << std::setw(4) << step_max-1 
      << std::endl;
    
    assemble_system(false);
    solve_system();

    Vector<double> solution_global(solution);
    
    output_results(
      this->output_dir, triangulation, 
      dof_handler, solution_global, mpi_comm, n);

    t += dt;
  }
  write_final_xdmf<dim>(this->output_dir, mpi_comm);
  
  pcout << "Completed." << std::endl;
}


template <int dim>
void NavierStokesGLS<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  locally_owned_dofs
   = dof_handler.locally_owned_dofs();
  locally_relevant_dofs 
  = DoFTools::extract_locally_relevant_dofs(dof_handler);

  DynamicSparsityPattern sparsity_pattern(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(
    dof_handler, sparsity_pattern);

  SparsityTools::distribute_sparsity_pattern(
    sparsity_pattern, locally_owned_dofs,
    mpi_comm, locally_relevant_dofs);

  system_matrix.reinit(
    locally_owned_dofs, locally_owned_dofs,
    sparsity_pattern, mpi_comm);
 
  system_rhs.reinit(locally_owned_dofs, mpi_comm);
  solution.reinit(locally_owned_dofs, mpi_comm);
  solution_prev.reinit(locally_owned_dofs, mpi_comm);
}

template <int dim>
std::array<double, 2> NavierStokesGLS<dim>::comp_stab_params_1(
  const Tensor<1, dim> &u, const double h)
{
  double vel_mag = 0e0;
  for (unsigned int d=0; d<dim; ++d)
    vel_mag += u[d]*u[d];
  vel_mag = std::sqrt(vel_mag);

  std::array<double, 2> tau;

  double Re = rho*vel_mag*h/mu;

  const double term1 = std::pow(2.0/dt, 2);
  const double term2 = std::pow(2.0*vel_mag/h, 2);
  const double term3 = std::pow(4.0*Re/(h*h), 2);
  
  tau[0] = 1e0/std::sqrt(term1 + term2 + term3);
  tau[1] = h*h/mu/12e0;
  // tau[1] = tau[0];

  return tau;
}

template <int dim>
std::array<double, 2> NavierStokesGLS<dim>::comp_stab_params_2(
  const Tensor<1, dim> &u, const double h)
{
  double vel_mag = 0.0;
  for (unsigned int d = 0; d < dim; ++d)
    vel_mag += u[d]*u[d];
  vel_mag = std::sqrt(vel_mag);

  double Re = rho*vel_mag*h/mu;

  const double beta[6] 
  = {1.0, 1.0/3.0, 30.0, 0.1, 1.0, 1.0};

  std::array<double, 2> tau;

  if (Re < 1.0e-10)
  {
    double fact = h*h/(4.0*mu/rho);
    tau[0] = fact*beta[0];
    tau[1] = fact*beta[2];
  }
  else
  {
    double fact2 = h/(2.0*vel_mag*rho);

    double fact = beta[0]/(beta[1]*Re);
    tau[0] = fact2*beta[0]/std::sqrt(1.0 + fact*fact);

    fact = beta[2]/(beta[3]*Re);
    tau[1] = fact2*beta[2]/std::sqrt(1.0 + fact*fact);
  }

  return tau;
}


template <int dim>
void NavierStokesGLS<dim>::assemble_matrix_initial(
  FullMatrix<double> &cell_matrix, FEValues<dim> &fe_values, 
  const double mu, std::array<double, 2> &tau, const unsigned int q)
{
  const auto &phi_u = fe_values[vel_ext];
  const auto &phi_p = fe_values[pre_ext];

  for (const unsigned int i : fe_values.dof_indices())
  {
    for (const unsigned int j : fe_values.dof_indices())
    {
      // diffusion term 
      cell_matrix(i, j) += mu*(
        scalar_product(phi_u.gradient(i, q), phi_u.gradient(j, q))
      )*fe_values.JxW(q);

      // pressure term
      cell_matrix(i, j) -= (
        phi_u.divergence(i, q)*phi_p.value(j, q)
      )*fe_values.JxW(q);

      // continuity term
      cell_matrix(i, j) += (
        phi_p.value(i, q)*phi_u.divergence(j, q) 
      )*fe_values.JxW(q);

      // PSPG pressure term
      cell_matrix(i, j) += tau[1]*(
        phi_p.gradient(i, q)*phi_p.gradient(j, q)
      )*fe_values.JxW(q);
    }
  }
}


template <int dim>
void NavierStokesGLS<dim>::assemble_matrix(
  FullMatrix<double> &cell_matrix, FEValues<dim> &fe_values, 
  std::array<double, 2> &tau, const unsigned int q)
{
  const auto &phi_u = fe_values[vel_ext];
  const auto &phi_p = fe_values[pre_ext];

  for (const unsigned int i : fe_values.dof_indices())
  {
    for (const unsigned int j : fe_values.dof_indices())
    {
      // mass term
      cell_matrix(i, j) += rho*(1e0/dt)*(
        phi_u.value(i, q)*phi_u.value(j, q)
      )*fe_values.JxW(q);

      // advection term 
      cell_matrix(i, j) += 5e-1*rho*(
        phi_u.value(i, q)*(phi_u.gradient(j, q)*u_q_adv[q])
      )*fe_values.JxW(q);

      // diffusion term 
      cell_matrix(i, j) += 5e-1*mu*(
        scalar_product(phi_u.gradient(i, q), phi_u.gradient(j, q))
      )*fe_values.JxW(q);

      // pressure term
      cell_matrix(i, j) -= (
        phi_u.divergence(i, q)*phi_p.value(j, q)
      )*fe_values.JxW(q);

      // continuity term
      cell_matrix(i, j) += (
        phi_p.value(i, q)*phi_u.divergence(j, q) 
      )*fe_values.JxW(q);

      // SUPG mass term 
      cell_matrix(i, j) += tau[0]*rho*(1e0/dt)*(
        (phi_u.gradient(i, q)*u_q_adv[q])*phi_u.value(j, q)
      )*fe_values.JxW(q);

      // SUPG advection term 
      cell_matrix(i, j) += 5e-1*tau[0]*rho*(
        (phi_u.gradient(i, q)*u_q_adv[q])*(phi_u.gradient(j, q)*u_q_adv[q])
      )*fe_values.JxW(q);

      // SUPG pressure term 
      cell_matrix(i, j) += tau[0]*(
        (phi_u.gradient(i, q)*u_q_adv[q])*phi_p.gradient(j, q)
      )*fe_values.JxW(q);

      // PSPG mass term
      cell_matrix(i, j) += tau[1]*rho*(1e0/dt)*(
        phi_p.gradient(i, q)*phi_u.value(j, q)
      )*fe_values.JxW(q);

      // PSPG advection term
      cell_matrix(i, j) += 5e-1*tau[1]*rho*(
        phi_p.gradient(i, q)*(phi_u.gradient(j, q)*u_q_adv[q])
      )*fe_values.JxW(q);

      // PSPG pressure term
      cell_matrix(i, j) += tau[1]*(
        phi_p.gradient(i, q)*phi_p.gradient(j, q)
      )*fe_values.JxW(q);
    }
  }

}


template <int dim>
void NavierStokesGLS<dim>::assemble_rhs(
  Vector<double> &cell_rhs, FEValues<dim> &fe_values, 
  std::array<double, 2> &tau, const unsigned int q)
{
  const auto &phi_u = fe_values[vel_ext];
  const auto &phi_p = fe_values[pre_ext];

  for (const unsigned int i : fe_values.dof_indices())
  {
    // mass term
    cell_rhs(i) += rho*(1e0/dt)*(
      phi_u.value(i, q)*u_q[q]
    )*fe_values.JxW(q);
    
    // advaction term
    cell_rhs(i) -= 5e-1*rho*(
      phi_u.value(i, q)*(u_q_grad[q]*u_q_adv[q])
    )*fe_values.JxW(q);

    // diffusion term
    cell_rhs(i) -= 5e-1*mu*(
      scalar_product(phi_u.gradient(i, q), u_q_grad[q])
    )*fe_values.JxW(q);

    // SUPG mass term 
    cell_rhs(i) += tau[0]*rho*(1e0/dt)*(
      (phi_u.gradient(i, q)*u_q_adv[q])*u_q[q]
    )*fe_values.JxW(q);

    // SUPG advection term 
    cell_rhs(i) -= 5e-1*tau[0]*rho*(
      (phi_u.gradient(i, q)*u_q_adv[q])*(u_q_grad[q]*u_q_adv[q])
    )*fe_values.JxW(q);

    // PSPG mass term
    cell_rhs(i) += tau[1]*(1e0/dt)*rho*(
      phi_p.gradient(i, q)*u_q[q]
    )*fe_values.JxW(q);

    // PSPG advection term
    cell_rhs(i) -= 5e-1*tau[1]*rho*(
      phi_p.gradient(i, q)*(u_q_grad[q]*u_q_adv[q])
    )*fe_values.JxW(q);
  }
}

template <int dim>
double NavierStokesGLS<dim>::comp_cell_length(
  FEValues<dim> &fe_values, Tensor<1, dim> &u, 
  const unsigned int dofs_per_cell, const unsigned int q)
{
  double length_sum = 0.0;

  const double u_mag = u.norm();
  Tensor<1, dim> u_norm;

  if (u_mag > 1e-10)
    u_norm = u/u_mag;
  else
    for (unsigned int d = 0; d < dim; ++d)
      u_norm[d] = 1.0/std::sqrt(dim);

  for (unsigned int i = 0; i < dofs_per_cell; ++i)
  {
    Tensor<1, dim> grad_phi = fe_values[pre_ext].gradient(i, q);
    length_sum += std::abs(u_norm*grad_phi);
  }

  return 2.0/length_sum;
}

template <int dim>
void NavierStokesGLS<dim>::apply_dirichlet_boundary_conditions(
  std::map<types::global_dof_index, double>& boundary_values)
{
  for (const auto &bc : bcs)
  {
    if (bc.dir == dim) 
    {
      VectorTools::interpolate_boundary_values(
        dof_handler,
        types::boundary_id(bc.id),
        VelocityUniform<dim>(bc.dir, bc.value),
        boundary_values,
        fe.component_mask(pre_ext)
      );
    }
    else
    {
      ComponentMask mask = fe.component_mask(vel_ext);
      for (unsigned int d = 0; d < mask.size(); ++d)
        mask.set(d, false);
      mask.set(bc.dir, true);

      VectorTools::interpolate_boundary_values(
        dof_handler,
        types::boundary_id(bc.id),
        VelocityUniform<dim>(bc.dir, bc.value),
        boundary_values,
        mask
      );
    }
  }
}

template <int dim>
void NavierStokesGLS<dim>::assemble_system(const bool initial)
{
  FEValues<dim> fe_values(
    fe, q_gauss,
    update_values | update_gradients | 
    update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points = q_gauss.size();
  
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  Vector<double> solution_global(solution);
  Vector<double> solution_global_prev(solution_prev);

  auto evaluate_values_on_q_points = [&]()
  {
    fe_values[vel_ext].get_function_values(solution_global, u_q);
    fe_values[vel_ext].get_function_values(solution_global_prev, u_q_prev);
    fe_values[vel_ext].get_function_gradients(solution_global, u_q_grad);
  };

  system_matrix = 0;
  system_rhs = 0;

  for (const auto& cell : dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      fe_values.reinit(cell);
      evaluate_values_on_q_points();

      cell_matrix = 0;
      cell_rhs = 0;
      
      for (unsigned int q = 0; q < n_q_points; ++q)
        u_q_adv[q] = 1.5*u_q[q] - 0.5*u_q_prev[q];

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        auto h = comp_cell_length(
          fe_values, u_q[q], dofs_per_cell, q);

        std::array<double, 2> tau
          = comp_stab_params_1(u_q[q], h);
        
        if (initial)
        {
          assemble_matrix_initial(
            cell_matrix, fe_values, mu, tau, q);
        }
        else
        {
          assemble_matrix(cell_matrix, fe_values, tau, q);
          assemble_rhs(cell_rhs, fe_values, tau, q);
        }
      }
      cell->get_dof_indices(local_dof_indices);
        
      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          system_matrix.add(local_dof_indices[i], 
                            local_dof_indices[j], 
                            cell_matrix(i, j));

      for (const unsigned int i : fe_values.dof_indices())
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }
  }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  std::map<types::global_dof_index, double> boundary_values;

  apply_dirichlet_boundary_conditions(boundary_values);

  // VectorTools::interpolate_boundary_values(
  //   dof_handler,
  //   types::boundary_id(this->inlet_label),
  //   VelocityUniform<dim>(0, 1e0),
  //   boundary_values,
  //   fe.component_mask(vel_ext)
  // );

  // VectorTools::interpolate_boundary_values(
  //   dof_handler,
  //   types::boundary_id(this->wall_label),
  //   WallVelocity<dim>(0.0),
  //   boundary_values,
  //   fe.component_mask(vel_ext)
  // );

  solution_prev = solution;

  MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution, system_rhs, false);
}

template <int dim>
void NavierStokesGLS<dim>::solve_system()
{    
  SolverControl solver_control(10000, 1e-6*system_rhs.l2_norm());
  PETScWrappers::SolverGMRES gmres(solver_control);

  PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);
  gmres.solve(system_matrix, solution, system_rhs, preconditioner);

  // return solver_control.last_step();
}
