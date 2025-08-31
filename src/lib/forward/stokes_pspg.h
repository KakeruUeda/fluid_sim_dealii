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
#include "runtime_params_stokes.h"

#include <string>
#include <sys/stat.h>

template <int dim>
class StokesPSPG : public PDEBase<dim>
{
public:
  using PDEBase<dim>::mpi_comm;
  using PDEBase<dim>::pcout;
  using PDEBase<dim>::fe;
  using PDEBase<dim>::triangulation;
  using PDEBase<dim>::dof_handler;
  using PDEBase<dim>::bcs;
  using PDEBase<dim>::make_grid;

  StokesPSPG(
    const RuntimeParams_Stokes &params);
  ~StokesPSPG() override = default;

  void run();

private:
  const double mu;

  std::string bc_data_path;
  BCData<dim> bc_data;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  const FEValuesExtractors::Vector vel_ext;
  const FEValuesExtractors::Scalar pre_ext;

  const QGaussSimplex <dim> q_gauss;

  PETScWrappers::MPI::SparseMatrix system_matrix;
  PETScWrappers::MPI::Vector system_rhs;
  PETScWrappers::MPI::Vector solution;

  void setup_system();
  void assemble_system();
  void solve_system();
  
  void assemble_matrix(
    FullMatrix<double> &cell_matrix, FEValues<dim> &fe_values, 
    const double tau, const unsigned int q);

  void apply_dirichlet_boundary_conditions(
    std::map<types::global_dof_index, double>& boundary_values);
};

template <int dim>
StokesPSPG<dim>::StokesPSPG(
    const RuntimeParams_Stokes &params)
    : PDEBase<dim>(params)
    , mu(params.mu)
    , bc_data_path(params.bc_data_path)
    , vel_ext(0)
    , pre_ext(dim)
    , q_gauss(2)
{
  std::string output_dir_tmp = "outputs";
  mkdir(output_dir_tmp.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);

  this->output_dir = "outputs/" + this->output_dir;
  mkdir(this->output_dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
}

template <int dim>
void StokesPSPG<dim>::run()
{
  make_grid();
  setup_system();

  process_bcs_from_h5(
    bc_data, pcout, bc_data_path
  );

  assemble_system();
  solve_system();

  Vector<double> solution_global(solution);
  
  output_results(
    this->output_dir, triangulation, 
    dof_handler, solution_global, mpi_comm, 0);

  write_xdmf_all<dim>(this->output_dir, mpi_comm);
  
  pcout << "Completed." << std::endl;
}

template <int dim>
void StokesPSPG<dim>::setup_system()
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
}

template <int dim>
void StokesPSPG<dim>::assemble_matrix(
  FullMatrix<double> &cell_matrix, FEValues<dim> &fe_values, 
  const double tau, const unsigned int q)
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
      cell_matrix(i, j) += tau*(
        phi_p.gradient(i, q)*phi_p.gradient(j, q)
      )*fe_values.JxW(q);
    }
  }
}

template <int dim>
void StokesPSPG<dim>::apply_dirichlet_boundary_conditions(
  std::map<types::global_dof_index, double>& boundary_values)
{
  for (const auto &bc : bc_data.bcs)
  {
    if (bc.dir == dim) // pressure
    {
      PressureUniformTimeSeries<dim> f(bc_data.time, bc.value);
      f.set_time(0);

      VectorTools::interpolate_boundary_values(
        dof_handler,
        types::boundary_id(bc.id),
        f,
        boundary_values,
        /*pressure mask*/ fe.component_mask(pre_ext)
      );
      continue;
    }

    ComponentMask one_comp(fe.n_components(), false);
    one_comp.set(bc.dir, true);

    if (bc.profile == ProfileType::Uniform)
    {
      VelocityUniformTimeSeries<dim> f(bc.dir, bc_data.time, bc.value);
      f.set_time(0);

      VectorTools::interpolate_boundary_values(
        dof_handler,
        types::boundary_id(bc.id),
        f,
        boundary_values,
        one_comp
      );
    }
    else if (bc.profile == ProfileType::Parabolic)
    {
      if (!bc.parabolic) continue;
      VelocityParabolicTimeSeries<dim> f(bc.dir, bc_data.time, bc.value, *bc.parabolic);
      f.set_time(0);

      VectorTools::interpolate_boundary_values(
        dof_handler,
        types::boundary_id(bc.id),
        f,
        boundary_values,
        one_comp
      );
    }
  }
  // for (const auto &bc : bcs)
  // {
  //   if (bc.dir == dim) 
  //   {
  //     VectorTools::interpolate_boundary_values(
  //       dof_handler,
  //       types::boundary_id(bc.id),
  //       VelocityUniform<dim>(bc.dir, bc.value),
  //       boundary_values,
  //       fe.component_mask(pre_ext)
  //     );
  //   }
  //   else
  //   {
  //     ComponentMask mask = fe.component_mask(vel_ext);
  //     for (unsigned int d = 0; d < mask.size(); ++d)
  //       mask.set(d, false);
  //     mask.set(bc.dir, true);

  //     VectorTools::interpolate_boundary_values(
  //       dof_handler,
  //       types::boundary_id(bc.id),
  //       VelocityUniform<dim>(bc.dir, bc.value),
  //       boundary_values,
  //       mask
  //     );
  //   }
  // }
}

template <int dim>
void StokesPSPG<dim>::assemble_system()
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

  system_matrix = 0;
  system_rhs = 0;

  for (const auto& cell : dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      fe_values.reinit(cell);

      cell_matrix = 0;
      cell_rhs = 0;
 
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        auto h = cell->diameter();
        const double tau = h*h/mu/12e0;
        assemble_matrix(cell_matrix, fe_values, tau, q);
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
  MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution, system_rhs, false);
}

template <int dim>
void StokesPSPG<dim>::solve_system()
{    
  SolverControl solver_control(10000, 1e-6*system_rhs.l2_norm());
  PETScWrappers::SolverGMRES gmres(solver_control);

  PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);
  gmres.solve(system_matrix, solution, system_rhs, preconditioner);

  // return solver_control.last_step();
}
