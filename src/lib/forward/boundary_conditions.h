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

enum class ProfileType
{
  Uniform,
  Parabolic
};

template <int dim>
struct ParabolicProfile
{
  double radius = 0.0;
  std::array<double, dim> center{{}};
};

template <int dim>
struct BoundaryCondition
{
  unsigned int id = 0;
  unsigned int dir = 0;
  std::vector<double> value;
  ProfileType profile = ProfileType::Uniform;
  std::optional<ParabolicProfile<dim>> parabolic;
};

template <int dim>
struct BCData
{
  std::vector<double> time;
  std::vector<BoundaryCondition<dim>> bcs;
};

inline double sample_series(const std::vector<double>& ts,
                            const std::vector<double>& vs, double t)
{
  if (ts.empty() || vs.empty() || ts.size()!=vs.size()) return 0.0;
  if (t <= ts.front()) return vs.front();
  if (t >= ts.back())  return vs.back();
  auto it = std::upper_bound(ts.begin(), ts.end(), t);
  size_t i1 = std::distance(ts.begin(), it);
  size_t i0 = i1 - 1;
  const double w = (t - ts[i0]) / (ts[i1] - ts[i0]);
  return (1.0 - w) * vs[i0] + w * vs[i1];
}

template <int dim>
class VelocityUniformTimeSeries : public Function<dim>
{
public:
  VelocityUniformTimeSeries(unsigned int dir,
                            std::vector<double> times,
                            std::vector<double> values)
    : Function<dim>(/*n_components=*/dim),
      dir_(dir), times_(std::move(times)), values_(std::move(values)) {}

  void vector_value(const Point<dim>&, Vector<double>& v) const override
  {
    if (v.size() != dim) v.reinit(dim);
    v = 0.0;
    if (dir_ >= dim) return;

    const double t = this->get_time();
    const double u = sample_series(times_, values_, t); 
    v[dir_] = u;
  }

private:
  unsigned int dir_;
  std::vector<double> times_;
  std::vector<double> values_;
};


template <int dim>
class VelocityParabolicTimeSeries : public Function<dim>
{
public:
  VelocityParabolicTimeSeries(unsigned int dir,
                              std::vector<double> times,
                              std::vector<double> values,
                              ParabolicProfile<dim> para)
    : Function<dim>(/*n_components=*/dim),
      dir_(dir), times_(std::move(times)), values_(std::move(values)),
      para_(std::move(para)) {}

  void vector_value(const Point<dim>& p, Vector<double>& v) const override
  {
    if (v.size() != dim) v.reinit(dim);
    v = 0.0;
    if (dir_ >= dim || para_.radius <= 0.0) return;

    double r2 = 0.0;
    for (unsigned d=0; d<dim; ++d) {
      const double d0 = p[d] - para_.center[d];
      r2 += d0*d0;
    }
    r2 = std::sqrt(r2);
    const double val = std::max(0.0, 1.0 - r2*r2/(para_.radius*para_.radius));
    const double t = this->get_time();
    const double u_max = sample_series(times_, values_, t);
    v[dir_] = u_max * val;
  }

private:
  unsigned int dir_;
  std::vector<double> times_;
  std::vector<double> values_;
  ParabolicProfile<dim> para_;
};


template <int dim>
class PressureUniformTimeSeries : public Function<dim>
{
public:
  PressureUniformTimeSeries(std::vector<double> times,
                            std::vector<double> values)
    : Function<dim>(/*n_components=*/dim+1),
      times_(std::move(times)), values_(std::move(values)) {}

  void vector_value(const Point<dim>&, Vector<double>& v) const override
  {
    if (v.size() != dim+1) v.reinit(dim+1);
    v = 0.0;
    v[dim] = sample_series(times_, values_, this->get_time());
  }
private:
  std::vector<double> times_, values_;
};

