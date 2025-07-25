/**
 * @file boundary_conditions.h
 * @author K.Ueda
 * @date Jun, 2025
 */

#pragma once
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/lac/vector.h>

using namespace dealii;

/**
 * @brief Struct representing a 
 * single boundary condition entry.
 *
 * Contains the boundary ID, 
 * direction, value, and type.
 */
struct BoundaryConditions
{
  unsigned int id;
  unsigned int dir;
  double value;
  std::string type;
};

/**
 * @brief Uniform inlet velocity boundary condition.
 *
 * The velocity is set to a constant value in the specified direction.
 * @tparam dim Spatial dimension.
 */
template <int dim>
class VelocityUniform : public Function<dim>
{
 public:
  /**
   * @brief Constructor.
   * @param dir Direction of the non-zero velocity component.
   * @param value Magnitude of the velocity.
   */
  VelocityUniform(const unsigned int dir, const double value);

  /**
   * @brief Evaluates the velocity vector at a given point.
   */
  virtual void vector_value(const Point<dim>&,
                            Vector<double>& v) const override;

 private:
  const unsigned int dir;
  const double value;
};

/**
 * @brief No-slip wall boundary condition (zero velocity).
 *
 * The velocity is always zero regardless of the input point.
 * @tparam dim Spatial dimension.
 */
template <int dim>
class WallVelocity : public Function<dim>
{
 public:
  /**
   * @brief Constructor.
   * @param value Not used (reserved for future extension).
   */
  explicit WallVelocity(const double value);

  /**
   * @brief Evaluates the velocity vector at a given point (should be
   * always zero).
   */
  virtual void vector_value(const Point<dim>&,
                            Vector<double>& v) const override;

 private:
  const double value;
};

// -------- Implementation --------

template <int dim>
VelocityUniform<dim>::VelocityUniform(
    const unsigned int dir, const double value)
    : Function<dim>(dim + 1), dir(dir), value(value)
{
}

template <int dim>
void VelocityUniform<dim>::vector_value(
  const Point<dim>&, Vector<double>& v) const
{
  v[dir] = value;
}

template <int dim>
WallVelocity<dim>::WallVelocity(const double value)
    : Function<dim>(dim + 1), value(value)
{
}

template <int dim>
void WallVelocity<dim>::vector_value(const Point<dim>&,
                                     Vector<double>& v) const
{
  v = 0;
}
