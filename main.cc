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
  std::cout << quadrature_formula.size() << std::endl;
  std::cout << quadrature_formula.point(2) << std::endl;

  // auto test = quadrature_formula.get_points();
  // std::cout << test.size() << std::endl;

  FEValues<dim> fe_values(fe, quadrature_formula, 
                          update_values | update_gradients | update_JxW_values);
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