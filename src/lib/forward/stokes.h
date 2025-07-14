#pragma once 
#include <pde_base.h>
#include <boundary_conditions.h>
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
  using PDEBase<dim>::make_grid;

  StokesPSPG(const RuntimeParams_Stokes &params);
  ~StokesPSPG() override = default;

  void run();

private:
  const double mu;
  
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  PETScWrappers::MPI::SparseMatrix system_matrix;
  PETScWrappers::MPI::Vector solution;
  PETScWrappers::MPI::Vector system_rhs;

  void setup_system();
  void assemble_system();
  unsigned int solve();
};

template <int dim>
StokesPSPG<dim>::StokesPSPG(
    const RuntimeParams_Stokes &params)
    : PDEBase<dim>(params)
    , mu(params.mu)
{
  std::string dir;
  std::string outputs_parent = "outputs";
  mkdir(outputs_parent.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);

  this->output_dir = "outputs/" + this->output_dir;
  mkdir(this->output_dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
}

template <int dim>
void StokesPSPG<dim>::run()
{
  make_grid();
  setup_system();
  
  pcout << "   Assembling..." 
        << std::endl;
  
  assemble_system();

  pcout << "   Solving..." 
        << std::endl << std::endl;
  const unsigned int n_iter = solve();

  pcout << "Solver converged in " 
        << n_iter<< " iterations." 
        << std::endl;

  output_results(this->output_dir, triangulation, 
                 dof_handler, solution, mpi_comm, 0);

  pcout << "Completed." << std::endl;
}


template <int dim>
void StokesPSPG<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  locally_owned_dofs = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
  
  DynamicSparsityPattern sparsity_pattern(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern);

  SparsityTools::distribute_sparsity_pattern(
    sparsity_pattern, locally_owned_dofs, mpi_comm, locally_relevant_dofs
  );

  system_matrix.reinit(
    locally_owned_dofs, locally_owned_dofs, sparsity_pattern, mpi_comm
  );
  system_rhs.reinit(locally_owned_dofs, mpi_comm);
  solution.reinit(locally_owned_dofs, mpi_comm);
}

template <int dim>
void StokesPSPG<dim>::assemble_system()
{
  const QGauss<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(
      fe, quadrature_formula,
      update_values | update_gradients | 
      update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points = quadrature_formula.size();
  
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  
  const FEValuesExtractors::Vector vel(0);
  const FEValuesExtractors::Scalar pre(dim);
  
  std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
  std::vector<double>  div_phi_u(dofs_per_cell);
  std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
  std::vector<double>  phi_p(dofs_per_cell);
  std::vector<Tensor<1, dim>> grad_phi_p(dofs_per_cell);

  for (const auto& cell : dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      fe_values.reinit(cell);

      cell_matrix = 0;
      cell_rhs = 0;

      const double h = cell->diameter();
      const double tau_pspg = h*h/mu/12e0;

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int k : fe_values.dof_indices())
        {
          grad_phi_u[k] = fe_values[vel].gradient(k, q);
          div_phi_u[k] = fe_values[vel].divergence(k, q);
          phi_u[k] = fe_values[vel].value(k, q);
          phi_p[k] = fe_values[pre].value(k, q);
          grad_phi_p[k] = fe_values[pre].gradient(k, q);
        }

        for (const unsigned int i : fe_values.dof_indices())
        {
          for (const unsigned int j : fe_values.dof_indices())
          {
            cell_matrix(i, j) += (
              - mu * scalar_product(grad_phi_u[i], grad_phi_u[j])   
              + div_phi_u[i] * phi_p[j]
              + phi_p[i] * div_phi_u[j]     
              + tau_pspg * grad_phi_p[i] * grad_phi_p[j]
            ) * fe_values.JxW(q);
          }
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

  VectorTools::interpolate_boundary_values(
    dof_handler,
    types::boundary_id(this->inlet_label),
    InletVelocityUniform<dim>(2, 1.0),
    boundary_values,
    fe.component_mask(vel)
  );

  VectorTools::interpolate_boundary_values(
    dof_handler,
    types::boundary_id(this->wall_label),
    WallVelocity<dim>(0.0),
    boundary_values,
    fe.component_mask(vel)
  );

  MatrixTools::apply_boundary_values(
    boundary_values, system_matrix, solution, system_rhs, false);
}


template <int dim>
unsigned int StokesPSPG<dim>::solve()
{
  SolverControl solver_control(1000, 1e-6 * system_rhs.l2_norm());
  PETScWrappers::SolverGMRES gmres(solver_control);

  PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);
  gmres.solve(system_matrix, solution, system_rhs, preconditioner);

  Vector<double> localized_solution(solution);
  solution = localized_solution;

  return solver_control.last_step();
}
