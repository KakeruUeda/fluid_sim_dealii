#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

using namespace dealii;

template <int dim>
void print_mesh_info(const Triangulation<dim>& triangulation, const std::string& filename)
{
  std::cout << "Mesh info:" << std::endl
            << " dimension: " << dim << std::endl
            << " no. of cells: " << triangulation.n_active_cells() << std::endl;

  {
    std::map<types::boundary_id, unsigned int> boundary_count;
    for (const auto& face : triangulation.active_face_iterators())
      if (face->at_boundary()) boundary_count[face->boundary_id()]++;

    std::cout << " boundary indicators: ";
    for (const auto& pair : boundary_count)
    {
      std::cout << pair.first << '(' << pair.second << " times) ";
    }
    std::cout << std::endl;
  }

  std::ofstream out(filename);
  GridOut grid_out;
  grid_out.write_vtu(triangulation, out);
  std::cout << " written to " << filename << std::endl << std::endl;
}

enum class BoundaryID {
  wall = 4,
  inlet = 5,
  outlet = 6 
};

template <int dim>
class NavierStokesProblem
{
public:
  NavierStokesProblem()
    : fe(1), dof_handler(triangulation){}
  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  Triangulation<dim> triangulation;
  const FE_SimplexP<dim> fe;
  DoFHandler<dim> dof_handler;
  SparsityPattern sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
};

template <int dim>
void NavierStokesProblem<dim>::assemble_system()
{
  const QGauss<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe, quadrature_formula, 
                          update_values | update_gradients | update_JxW_values);
  
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();   

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);

    cell_matrix = 0;
    cell_rhs = 0;

    for (const unsigned int q_index : fe_values.quadrature_point_indices()) 
    {
      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          cell_matrix(i, j) +=
            (fe_values.shape_grad(i, q_index) *
             fe_values.shape_grad(j, q_index) *
             fe_values.JxW(q_index));

      for (const unsigned int i : fe_values.dof_indices())
        cell_rhs(i) += (fe_values.shape_value(i, q_index)) *
                        1. * 
                        fe_values.JxW(q_index);
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
  std::map<types::global_dof_index, double> boundary_values;

  // VectorTools::interpolate_boundary_values(dof_handler, 
  //                                          types::boundary_id(BoundaryID::wall), 
  //                                          Functions::ZeroFunction<dim>(),
  //                                          boundary_values);
  VectorTools::interpolate_boundary_values(dof_handler, 
                                           types::boundary_id(BoundaryID::inlet), 
                                           Functions::ZeroFunction<dim>(),
                                           boundary_values);
  // VectorTools::interpolate_boundary_values(dof_handler, 
  //                                          types::boundary_id(BoundaryID::outlet), 
  //                                          Functions::ZeroFunction<dim>(),
  //                                          boundary_values);
                                           
  std::cout << boundary_values.size() << std::endl;

  MatrixTools::apply_boundary_values(boundary_values, 
                                     system_matrix,
                                     solution,
                                     system_rhs);
}

template <int dim>
void NavierStokesProblem<dim>::solve()
{
  SolverControl            solver_control(1000, 1e-6 * system_rhs.l2_norm());
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
 
  std::cout << solver_control.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;
}
 
 
template <int dim>
void NavierStokesProblem<dim>::output_results() const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
 
  const std::string filename = "solution.vtk";
  std::ofstream output(filename);
  data_out.write_vtk(output);
  std::cout << "Output written to " << filename << std::endl;
}
 
 

template <int dim>
void NavierStokesProblem<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  std::cout << "Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}

template <int dim>
void NavierStokesProblem<dim>::make_grid()
{
  GridIn<dim> gridin;
  gridin.attach_triangulation(triangulation);

  std::ifstream f("../mesh/test.msh");
  gridin.read_msh(f);

  print_mesh_info(triangulation, "grid.vtu");
}

template <int dim> 
void NavierStokesProblem<dim>::run()
{
  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();
}


int main()
{
  try
  {
    NavierStokesProblem<3> navier_stokes;
    navier_stokes.run();
  }
  catch (std::exception& exc)
  {
    std::cerr << "Exeption on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting .." << std::endl;
  }
}