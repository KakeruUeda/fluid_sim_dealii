#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/grid/grid_tools.h>

#include <fstream>
#include <iostream>

#include "boundary_conditions.h"
#include "io.h"
#include "runtime_params_base.h"

using namespace dealii;

template <int dim>
class PDEBase
{
public:
  PDEBase(const RuntimeParams &params);
  virtual ~PDEBase();

protected:
  MPI_Comm mpi_comm;
  const unsigned int n_mpi_proc;
  const unsigned int this_mpi_proc;

  std::string output_dir;
  const std::string mesh_dir;

  const unsigned int inlet_label;
  const unsigned int outlet_label;
  const unsigned int wall_label;

  const std::vector<BoundaryConditions> bcs;

  parallel::shared::Triangulation<dim> triangulation;
  const FESystem<dim> fe;
  DoFHandler<dim> dof_handler;
  
  ConditionalOStream pcout;

  void make_grid();
};

template <int dim>
PDEBase<dim>::PDEBase(const RuntimeParams &params)
  : mpi_comm(MPI_COMM_WORLD)
  , n_mpi_proc(Utilities::MPI::n_mpi_processes(mpi_comm))
  , this_mpi_proc(Utilities::MPI::this_mpi_process(mpi_comm))
  , output_dir(params.output_dir)
  , mesh_dir(params.mesh_dir)
  , inlet_label(params.inlet_label)
  , outlet_label(params.outlet_label)
  , wall_label(params.wall_label)
  , bcs(params.bcs)
  , triangulation(MPI_COMM_WORLD)
  , fe(FE_SimplexP<dim>(params.degree_vel)^dim,
       FE_SimplexP<dim>(params.degree_pre))
  , dof_handler(triangulation)
  , pcout(std::cout, (this_mpi_proc == 0))
{
}

// -------- Implementation --------

template <int dim>
PDEBase<dim>::~PDEBase()
{
  dof_handler.clear();
}

template <int dim>
void PDEBase<dim>::make_grid()
{
  GridIn<dim> gridin;
  gridin.attach_triangulation(triangulation);

  std::ifstream f(mesh_dir);
  gridin.read_msh(f);

  print_mesh_info(triangulation, pcout);

  bool output_grid = true;
  if (output_grid && this_mpi_proc == 0)
  {
    std::ofstream out(output_dir + "/grid.vtu");
    GridOut grid_out;
    grid_out.write_vtu(triangulation, out);
    pcout << " written to " << "grid.vtu" << std::endl << std::endl;
  }
}

