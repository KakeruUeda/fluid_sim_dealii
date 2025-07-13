#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

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

using namespace dealii;

template <int dim>
void print_mesh_info(const Triangulation<dim>& triangulation, const std::string& filename,
                     ConditionalOStream& pcout)
{
  pcout << "Mesh info:" << std::endl
        << " dimension: " << dim << std::endl
        << " no. of cells: " << triangulation.n_active_cells() << std::endl;

  {
    std::map<types::boundary_id, unsigned int> boundary_count;
    for (const auto& face : triangulation.active_face_iterators())
      if (face->at_boundary()) boundary_count[face->boundary_id()]++;

    pcout << " boundary indicators: ";
    for (const auto& pair : boundary_count)
    {
      pcout << pair.first << '(' << pair.second << " times) ";
    }
    pcout << std::endl;
  }

  std::ofstream out(filename);
  GridOut grid_out;
  grid_out.write_vtu(triangulation, out);
  pcout << " written to " << filename << std::endl << std::endl;
}

enum class BoundaryID
{
  wall = 4,
  inlet = 5,
  outlet = 6
};

template <int dim>
class NavierStokesProblem
{
 public:
  NavierStokesProblem();
  ~NavierStokesProblem();
  void run();

 private:
  void make_grid();
  void setup_system();
  void assemble_system();
  unsigned int solve();
  void output_results() const;

  MPI_Comm mpi_comm;
  const unsigned int n_mpi_proc;
  const unsigned int this_mpi_proc;

  parallel::shared::Triangulation<dim> triangulation;
  const FESystem<dim> fe;
  DoFHandler<dim> dof_handler;
  
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  ConditionalOStream pcout;

  PETScWrappers::MPI::SparseMatrix system_matrix;
  PETScWrappers::MPI::Vector solution;
  PETScWrappers::MPI::Vector system_rhs;
};

template <int dim>
NavierStokesProblem<dim>::NavierStokesProblem()
  : mpi_comm(MPI_COMM_WORLD),
  n_mpi_proc(Utilities::MPI::n_mpi_processes(mpi_comm)),
  this_mpi_proc(Utilities::MPI::this_mpi_process(mpi_comm)),
  triangulation(MPI_COMM_WORLD),
  fe(FE_SimplexP<dim>(1) ^ dim, FE_SimplexP<dim>(1)),
  dof_handler(triangulation),
  pcout(std::cout, (this_mpi_proc == 0))
{}

template <int dim>
NavierStokesProblem<dim>::~NavierStokesProblem()
{
  dof_handler.clear();
}

template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  BoundaryValues();

  virtual double value(
    const Point<dim>& p, 
    const unsigned int component = 0
  ) const override;
};

template<int dim>
BoundaryValues<dim>::BoundaryValues()
  : Function<dim>(dim+1)
{}

template <int dim>
double BoundaryValues<dim>::value(
  const Point<dim>&, const unsigned int) const
{
  return 1.0;
}

template <int dim>
class InletVelocity : public Function<dim>
{
public:
  InletVelocity() : Function<dim>(dim+1) {}
  virtual void vector_value(const Point<dim> &, Vector<double> &v) const override
  {
    v = 0;
    v[2] = 1.0; 
  }
};

template <int dim>
class WallVelocity : public Function<dim>
{
public:
  WallVelocity() : Function<dim>(dim+1) {}
  virtual void vector_value(const Point<dim> &, Vector<double> &v) const override
  {
    v = 0;
  }
};


template <int dim>
void NavierStokesProblem<dim>::assemble_system()
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

  const double mu = 1.0;

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
    types::boundary_id(BoundaryID::inlet),
    InletVelocity<dim>(),
    boundary_values,
    fe.component_mask(vel)
  );

  VectorTools::interpolate_boundary_values(
    dof_handler,
    types::boundary_id(BoundaryID::wall),
    WallVelocity<dim>(),
    boundary_values,
    fe.component_mask(vel)
  );

  MatrixTools::apply_boundary_values(
    boundary_values, system_matrix, solution, system_rhs, false);
}

template <int dim>
unsigned int NavierStokesProblem<dim>::solve()
{
  SolverControl solver_control(1000, 1e-6 * system_rhs.l2_norm());
  PETScWrappers::SolverGMRES gmres(solver_control);

  PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);
  gmres.solve(system_matrix, solution, system_rhs, preconditioner);

  Vector<double> localized_solution(solution);
  solution = localized_solution;

  return solver_control.last_step();
}


template <int dim>
void NavierStokesProblem<dim>::output_results() const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  std::vector<std::string> solution_names(dim, "velocity");
  solution_names.emplace_back("pressure");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  interpretation.push_back(DataComponentInterpretation::component_is_scalar);

  data_out.add_data_vector(
    solution, solution_names,
    DataOut<dim>::type_dof_data,
    interpretation);

  std::vector<unsigned int> partition_int(triangulation.n_active_cells());
  GridTools::get_subdomain_association(triangulation, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  DataOutBase::DataOutFilterFlags flags(true, true);
  DataOutBase::DataOutFilter data_filter(flags);

  data_out.write_filtered_data(data_filter);
  data_out.write_hdf5_parallel(data_filter, "solution.h5", mpi_comm);

  auto new_xdmf_entry = data_out.create_xdmf_entry(
      data_filter, "solution.h5", 0.0, mpi_comm);
  std::vector<XDMFEntry> xdmf_entries;
  xdmf_entries.push_back(new_xdmf_entry);
  data_out.write_xdmf_file(xdmf_entries, "solution.xdmf", mpi_comm);
}


template <int dim>
void NavierStokesProblem<dim>::setup_system()
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
void NavierStokesProblem<dim>::make_grid()
{
  GridIn<dim> gridin;
  gridin.attach_triangulation(triangulation);

  std::ifstream f("../mesh/test.msh");
  gridin.read_msh(f);

  print_mesh_info(triangulation, "grid.vtu", pcout);
}

template <int dim>
void NavierStokesProblem<dim>::run()
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

  output_results();

  pcout << "Completed." << std::endl;
}

using MyMPI = Utilities::MPI::MPI_InitFinalize;

int main(int argc, char* argv[])
{
  try
  {
    MyMPI mpi_init(argc, argv, 1);

    NavierStokesProblem<3> navier_stokes;
    navier_stokes.run();
  }
  catch (std::exception& exc)
  {
    std::cerr << "Exeption on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting .." << std::endl;
    return 1;
  }
}