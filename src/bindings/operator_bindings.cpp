// Copyright 2026 Sebastien Dubois (sbstndbs)
// SPDX-License-Identifier: Apache-2.0

// Samurai Python Bindings - Operator functions
//
// Bindings for finite volume operators like upwind, convection_weno5

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <samurai/algorithm.hpp>
#include <samurai/algorithm/update.hpp>
#include <samurai/field.hpp>
#include <samurai/mesh_config.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/schemes/fv/explicit_FV_scheme.hpp>
#include <samurai/schemes/fv/flux_based/explicit_flux_based_scheme__lin_hom.hpp>
#include <samurai/schemes/fv/flux_based/explicit_flux_based_scheme__nonlin.hpp>
#include <samurai/schemes/fv/operators/convection_lin.hpp>
#include <samurai/schemes/fv/operators/convection_nonlin.hpp>
#include <samurai/schemes/fv/operators/diffusion.hpp>
#include <samurai/schemes/fv/operators/divergence.hpp>
#include <samurai/schemes/fv/operators/gradient.hpp>
#include <samurai/stencil_field.hpp>
#include "common_types.hpp"

namespace py = pybind11;

// Use centralized type aliases from common_types.hpp
using namespace samurai::python::bindings;

// Note: operator_bindings.cpp uses Interval<int, long long int> for algorithms
// which differs from the algorithm_interval in common_types.hpp (Interval<double, std::size_t>)
// This is intentional for algorithm-specific use cases
using algorithm_interval = samurai::Interval<int, long long int>;

// 1D upwind operator - with optional output field (no allocation when provided)
py::object upwind_1d(ScalarField<1>& field, double velocity, py::object output_obj = py::none())
{
    auto& mesh = field.mesh();

    // Get the upwind expression
    auto upwind_expr = samurai::upwind(velocity, field);

    if (!output_obj.is_none())
    {
        // No allocation: write directly to provided output field
        auto& output = output_obj.cast<ScalarField<1>&>();
        samurai::for_each_interval(mesh,
                                   [&output, &upwind_expr](std::size_t level, const algorithm_interval& interval, const auto& index)
                                   {
                                       output(level, interval, index) = upwind_expr(level, interval, index);
                                   });
        return output_obj;  // Return the output field for chaining
    }
    else
    {
        // Backward compatible: allocate and return new field
        auto result = samurai::make_scalar_field<double>(field.name() + "_upwind", mesh);
        samurai::for_each_interval(mesh,
                                   [&result, &upwind_expr](std::size_t level, const algorithm_interval& interval, const auto& index)
                                   {
                                       result(level, interval, index) = upwind_expr(level, interval, index);
                                   });
        return py::cast(result);
    }
}

// 2D upwind operator - with optional output field (no allocation when provided)
py::object upwind_2d(ScalarField<2>& field, const std::array<double, 2>& velocity, py::object output_obj = py::none())
{
    auto& mesh = field.mesh();

    // Get the upwind expression
    auto upwind_expr = samurai::upwind(velocity, field);

    if (!output_obj.is_none())
    {
        // No allocation: write directly to provided output field
        auto& output = output_obj.cast<ScalarField<2>&>();
        samurai::for_each_interval(mesh,
                                   [&output, &upwind_expr](std::size_t level, const algorithm_interval& interval, const auto& index)
                                   {
                                       output(level, interval, index) = upwind_expr(level, interval, index);
                                   });
        return output_obj;  // Return the output field for chaining
    }
    else
    {
        // Backward compatible: allocate and return new field
        auto result = samurai::make_scalar_field<double>(field.name() + "_upwind", mesh);
        samurai::for_each_interval(mesh,
                                   [&result, &upwind_expr](std::size_t level, const algorithm_interval& interval, const auto& index)
                                   {
                                       result(level, interval, index) = upwind_expr(level, interval, index);
                                   });
        return py::cast(result);
    }
}

// 3D upwind operator - with optional output field (no allocation when provided)
py::object upwind_3d(ScalarField<3>& field, const std::array<double, 3>& velocity, py::object output_obj = py::none())
{
    auto& mesh = field.mesh();

    // Get the upwind expression
    auto upwind_expr = samurai::upwind(velocity, field);

    if (!output_obj.is_none())
    {
        // No allocation: write directly to provided output field
        auto& output = output_obj.cast<ScalarField<3>&>();
        samurai::for_each_interval(mesh,
                                   [&output, &upwind_expr](std::size_t level, const algorithm_interval& interval, const auto& index)
                                   {
                                       output(level, interval, index) = upwind_expr(level, interval, index);
                                   });
        return output_obj;  // Return the output field for chaining
    }
    else
    {
        // Backward compatible: allocate and return new field
        auto result = samurai::make_scalar_field<double>(field.name() + "_upwind", mesh);
        samurai::for_each_interval(mesh,
                                   [&result, &upwind_expr](std::size_t level, const algorithm_interval& interval, const auto& index)
                                   {
                                       result(level, interval, index) = upwind_expr(level, interval, index);
                                   });
        return py::cast(result);
    }
}

// Convenience wrapper accepting Python list/tuple for velocity (2D)
py::object upwind_2d_py(ScalarField<2>& field, py::sequence velocity_seq, py::object output_obj = py::none())
{
    if (len(velocity_seq) != 2)
    {
        throw std::runtime_error("Velocity must have exactly 2 elements for 2D");
    }

    std::array<double, 2> velocity;
    velocity[0] = velocity_seq[0].cast<double>();
    velocity[1] = velocity_seq[1].cast<double>();

    return upwind_2d(field, velocity, output_obj);
}

// Convenience wrapper accepting Python list/tuple for velocity (3D)
py::object upwind_3d_py(ScalarField<3>& field, py::sequence velocity_seq, py::object output_obj = py::none())
{
    if (len(velocity_seq) != 3)
    {
        throw std::runtime_error("Velocity must have exactly 3 elements for 3D");
    }

    std::array<double, 3> velocity;
    velocity[0] = velocity_seq[0].cast<double>();
    velocity[1] = velocity_seq[1].cast<double>();
    velocity[2] = velocity_seq[2].cast<double>();

    return upwind_3d(field, velocity, output_obj);
}

// -------------------------------------------------------------------------
// WENO5 Convection operators (5th order Weighted Essentially Non-Oscillatory)
// -------------------------------------------------------------------------

// ============================================================
// Non-linear WENO5 (for Burgers equation): f(u) = u^2 or u(d)*u
// ============================================================

// 1D non-linear WENO5 (scalar Burgers)
py::object convection_weno5_nonlin_1d(ScalarField<1>& field)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_conv", mesh);

    // Create WENO5 convection operator (nonlinear)
    auto conv = samurai::make_convection_weno5<std::decay_t<decltype(field)>>();

    // Get the expression and evaluate it
    auto conv_expr = conv(field);
    result         = conv_expr;

    return py::cast(result);
}

// 2D non-linear WENO5 (scalar Burgers)
py::object convection_weno5_nonlin_2d(ScalarField<2>& field)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_conv", mesh);

    // Create WENO5 convection operator (nonlinear)
    auto conv = samurai::make_convection_weno5<std::decay_t<decltype(field)>>();

    // Get the expression and evaluate it
    auto conv_expr = conv(field);
    result         = conv_expr;

    return py::cast(result);
}

// 3D non-linear WENO5 (scalar Burgers)
py::object convection_weno5_nonlin_3d(ScalarField<3>& field)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_conv", mesh);

    // Create WENO5 convection operator (nonlinear)
    auto conv = samurai::make_convection_weno5<std::decay_t<decltype(field)>>();

    // Get the expression and evaluate it
    auto conv_expr = conv(field);
    result         = conv_expr;

    return py::cast(result);
}

// ============================================================
// Linear WENO5 with constant velocity: f(u) = velocity * u
// ============================================================

// 1D linear WENO5 with constant velocity
py::object convection_weno5_linear_1d(ScalarField<1>& field, double velocity)
{
    using VelocityVector = samurai::VelocityVector<1>;
    auto& mesh           = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_conv", mesh);

    // Create velocity vector
    VelocityVector vel;
    vel(0) = velocity;

    // Create WENO5 convection operator (linear with constant velocity)
    auto conv = samurai::make_convection_weno5<std::decay_t<decltype(field)>>(vel);

    // Get the expression and evaluate it
    auto conv_expr = conv(field);
    result         = conv_expr;

    return py::cast(result);
}

// 2D linear WENO5 with constant velocity
py::object convection_weno5_linear_2d(ScalarField<2>& field, const std::array<double, 2>& velocity)
{
    using VelocityVector = samurai::VelocityVector<2>;
    auto& mesh           = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_conv", mesh);

    // Create velocity vector
    VelocityVector vel;
    vel(0) = velocity[0];
    vel(1) = velocity[1];

    // Create WENO5 convection operator (linear with constant velocity)
    auto conv = samurai::make_convection_weno5<std::decay_t<decltype(field)>>(vel);

    // Get the expression and evaluate it
    auto conv_expr = conv(field);
    result         = conv_expr;

    return py::cast(result);
}

// 2D linear WENO5 - Python sequence version
py::object convection_weno5_linear_2d_py(ScalarField<2>& field, py::sequence velocity_seq)
{
    if (len(velocity_seq) != 2)
    {
        throw std::runtime_error("Velocity must have exactly 2 elements for 2D");
    }

    std::array<double, 2> velocity;
    velocity[0] = velocity_seq[0].cast<double>();
    velocity[1] = velocity_seq[1].cast<double>();

    return convection_weno5_linear_2d(field, velocity);
}

// 3D linear WENO5 with constant velocity
py::object convection_weno5_linear_3d(ScalarField<3>& field, const std::array<double, 3>& velocity)
{
    using VelocityVector = samurai::VelocityVector<3>;
    auto& mesh           = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_conv", mesh);

    // Create velocity vector
    VelocityVector vel;
    vel(0) = velocity[0];
    vel(1) = velocity[1];
    vel(2) = velocity[2];

    // Create WENO5 convection operator (linear with constant velocity)
    auto conv = samurai::make_convection_weno5<std::decay_t<decltype(field)>>(vel);

    // Get the expression and evaluate it
    auto conv_expr = conv(field);
    result         = conv_expr;

    return py::cast(result);
}

// 3D linear WENO5 - Python sequence version
py::object convection_weno5_linear_3d_py(ScalarField<3>& field, py::sequence velocity_seq)
{
    if (len(velocity_seq) != 3)
    {
        throw std::runtime_error("Velocity must have exactly 3 elements for 3D");
    }

    std::array<double, 3> velocity;
    velocity[0] = velocity_seq[0].cast<double>();
    velocity[1] = velocity_seq[1].cast<double>();
    velocity[2] = velocity_seq[2].cast<double>();

    return convection_weno5_linear_3d(field, velocity);
}

// ============================================================
// Non-linear WENO5 for VectorField (Burgers equation): f(u) = u(d)*u
// ============================================================

// 2D non-linear WENO5 for VectorField2D_2 (Burgers 2D)
py::object convection_weno5_nonlin_vector_2d(VectorField2D_2& field)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_vector_field<double, 2, false>(field.name() + "_conv", mesh);

    // Create WENO5 convection operator (nonlinear for vector field)
    auto conv = samurai::make_convection_weno5<std::decay_t<decltype(field)>>();

    // Get the expression and evaluate it
    auto conv_expr = conv(field);
    result         = conv_expr;

    return py::cast(result);
}

// 3D non-linear WENO5 for VectorField3D_3 (Burgers 3D)
py::object convection_weno5_nonlin_vector_3d(VectorField3D_3& field)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_vector_field<double, 3, false>(field.name() + "_conv", mesh);

    // Create WENO5 convection operator (nonlinear for vector field)
    auto conv = samurai::make_convection_weno5<std::decay_t<decltype(field)>>();

    // Get the expression and evaluate it
    auto conv_expr = conv(field);
    result         = conv_expr;

    return py::cast(result);
}

// ============================================================
// Linear WENO5 with VectorField velocity: f(u) = velocity(x) * u
// For convection with spatially varying velocity fields
// ============================================================

// 2D ScalarField with VectorField2D_2 velocity
py::object convection_weno5_vectorfield_2d(ScalarField<2>& field, VectorField2D_2& velocity)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_conv", mesh);

    // Create WENO5 convection operator with VectorField velocity
    // Uses template deduction to select the correct overload
    auto conv = samurai::make_convection_weno5<std::decay_t<decltype(field)>>(velocity);

    // Get the expression and evaluate it
    auto conv_expr = conv(field);
    result         = conv_expr;

    return py::cast(result);
}

// 3D ScalarField with VectorField3D_3 velocity
py::object convection_weno5_vectorfield_3d(ScalarField<3>& field, VectorField3D_3& velocity)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_conv", mesh);

    // Create WENO5 convection operator with VectorField velocity
    auto conv = samurai::make_convection_weno5<std::decay_t<decltype(field)>>(velocity);

    // Get the expression and evaluate it
    auto conv_expr = conv(field);
    result         = conv_expr;

    return py::cast(result);
}

// ============================================================
// Diffusion operators (order 2)
// ============================================================
// Diffusion operator: -coeff * Laplacian
// Note: diffusion_order2 implements -K*Delta (negative Laplacian)

py::object diffusion_order2_1d(ScalarField<1>& field, double coefficient = 1.0)
{
    auto& mesh = field.mesh();

    // Create diffusion operator with coefficient
    auto diff = samurai::make_diffusion_order2<ScalarField<1>>(coefficient);

    // Create output field
    auto result = samurai::make_scalar_field<double>(field.name() + "_diff", mesh);

    // Apply operator
    result = diff(field);

    return py::cast(result);
}

py::object diffusion_order2_2d(ScalarField<2>& field, double coefficient = 1.0)
{
    auto& mesh = field.mesh();

    // Create diffusion operator with coefficient
    auto diff = samurai::make_diffusion_order2<ScalarField<2>>(coefficient);

    // Create output field
    auto result = samurai::make_scalar_field<double>(field.name() + "_diff", mesh);

    // Apply operator
    result = diff(field);

    return py::cast(result);
}

py::object diffusion_order2_3d(ScalarField<3>& field, double coefficient = 1.0)
{
    auto& mesh = field.mesh();

    // Create diffusion operator with coefficient
    auto diff = samurai::make_diffusion_order2<ScalarField<3>>(coefficient);

    // Create output field
    auto result = samurai::make_scalar_field<double>(field.name() + "_diff", mesh);

    // Apply operator
    result = diff(field);

    return py::cast(result);
}

// ============================================================
// Laplacian operators (order 2)
// ============================================================
// Laplacian operator: Delta (positive Laplacian)
// Note: laplacian = -diffusion (with coefficient = 1)

py::object laplacian_order2_1d(ScalarField<1>& field)
{
    auto& mesh = field.mesh();

    // Laplacian = -Diffusion (with K=1). Use diffusion with coefficient -1 to get positive Laplacian
    // Since diffusion computes -K*Laplacian, using K=-1 gives -(-1)*Laplacian = Laplacian
    auto diff = samurai::make_diffusion_order2<ScalarField<1>>(-1.0);

    // Create output field
    auto result = samurai::make_scalar_field<double>(field.name() + "_laplacian", mesh);

    // Apply operator
    result = diff(field);

    return py::cast(result);
}

py::object laplacian_order2_2d(ScalarField<2>& field)
{
    auto& mesh = field.mesh();

    // Laplacian = -Diffusion (with K=1). Use diffusion with coefficient -1 to get positive Laplacian
    auto diff = samurai::make_diffusion_order2<ScalarField<2>>(-1.0);

    // Create output field
    auto result = samurai::make_scalar_field<double>(field.name() + "_laplacian", mesh);

    // Apply operator
    result = diff(field);

    return py::cast(result);
}

py::object laplacian_order2_3d(ScalarField<3>& field)
{
    auto& mesh = field.mesh();

    // Laplacian = -Diffusion (with K=1). Use diffusion with coefficient -1 to get positive Laplacian
    auto diff = samurai::make_diffusion_order2<ScalarField<3>>(-1.0);

    // Create output field
    auto result = samurai::make_scalar_field<double>(field.name() + "_laplacian", mesh);

    // Apply operator
    result = diff(field);

    return py::cast(result);
}

// ============================================================
// Gradient operators (order 2)
// ============================================================
// Gradient operator: returns VectorField<dim, dim> from ScalarField<dim>
// NOTE: 1D gradient is not supported due to VectorField1D_1 incompatibility with explicit scheme

py::object gradient_order2_2d(ScalarField<2>& field)
{
    auto& mesh = field.mesh();

    // Create gradient operator
    auto grad = samurai::make_gradient_order2<ScalarField<2>>();

    // Create output field (VectorField2D_2)
    auto result = samurai::make_vector_field<double, 2, false>(field.name() + "_grad", mesh);

    // Apply operator
    result = grad(field);

    return py::cast(result);
}

py::object gradient_order2_3d(ScalarField<3>& field)
{
    auto& mesh = field.mesh();

    // Create gradient operator
    auto grad = samurai::make_gradient_order2<ScalarField<3>>();

    // Create output field (VectorField3D_3)
    auto result = samurai::make_vector_field<double, 3, false>(field.name() + "_grad", mesh);

    // Apply operator
    result = grad(field);

    return py::cast(result);
}

// ============================================================
// Divergence operators (order 2)
// ============================================================
// Divergence operator: returns ScalarField<dim> from VectorField<dim, dim>
// NOTE: 1D divergence is not supported due to VectorField1D_1 incompatibility with explicit scheme

py::object divergence_order2_2d(VectorField2D_2& field)
{
    auto& mesh = field.mesh();

    // Create divergence operator
    auto div = samurai::make_divergence_order2<VectorField2D_2>();

    // Create output field (ScalarField2D)
    auto result = samurai::make_scalar_field<double>(field.name() + "_div", mesh);

    // Apply operator
    result = div(field);

    return py::cast(result);
}

py::object divergence_order2_3d(VectorField3D_3& field)
{
    auto& mesh = field.mesh();

    // Create divergence operator
    auto div = samurai::make_divergence_order2<VectorField3D_3>();

    // Create output field (ScalarField3D)
    auto result = samurai::make_scalar_field<double>(field.name() + "_div", mesh);

    // Apply operator
    result = div(field);

    return py::cast(result);
}

// Module initialization function for operator bindings
void init_operator_bindings(py::module_& m)
{
    // ============================================================
    // Upwind operators (with optional output field for no allocation)
    // ============================================================

    // Bind 1D upwind operator
    m.def("upwind",
          &upwind_1d,
          py::arg("field"),
          py::arg("velocity"),
          py::arg("output") = py::none(),
          R"pbdoc(
        Upwind operator for 1D advection.

        Computes the upwind flux for a scalar field in 1D.

        Parameters
        ----------
        field : ScalarField1D
            Input scalar field
        velocity : float
            Advection velocity (scalar for 1D)
        output : ScalarField1D, optional
            Pre-allocated output field for no-allocation mode.
            If provided, result is written directly to this field.
            If not provided, a new field is allocated and returned.

        Returns
        -------
        ScalarField1D
            Field containing upwind flux values (either output or newly allocated)

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh1D(box, config)
        >>> u = sam.ScalarField1D("u", mesh)
        >>> # Allocating version (backward compatible):
        >>> flux = sam.upwind(u, 1.0)
        >>> # No-allocation version (efficient):
        >>> flux = sam.ScalarField1D("flux", mesh, 0.0)
        >>> sam.upwind(u, 1.0, output=flux)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc");

    // Bind 2D upwind operator - std::array version
    m.def("upwind",
          &upwind_2d,
          py::arg("field"),
          py::arg("velocity"),
          py::arg("output") = py::none(),
          R"pbdoc(
        Upwind operator for 2D advection (std::array version).

        Parameters
        ----------
        field : ScalarField2D
            Input scalar field
        velocity : std::array<double, 2>
            2D velocity vector [vx, vy]
        output : ScalarField2D, optional
            Pre-allocated output field for no-allocation mode

        Returns
        -------
        ScalarField2D
            Field containing upwind flux values
        )pbdoc");

    // Bind 2D upwind operator - Python sequence version (more convenient)
    m.def("upwind",
          &upwind_2d_py,
          py::arg("field"),
          py::arg("velocity"),
          py::arg("output") = py::none(),
          R"pbdoc(
        Upwind operator for 2D advection.

        Computes the upwind flux for a scalar field in 2D.

        Parameters
        ----------
        field : ScalarField2D
            Input scalar field
        velocity : sequence of float
            2D velocity vector [vx, vy] (list or tuple)
        output : ScalarField2D, optional
            Pre-allocated output field for no-allocation mode

        Returns
        -------
        ScalarField2D
            Field containing upwind flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh2D(box, config)
        >>> u = sam.ScalarField2D("u", mesh)
        >>> velocity = [1.0, 1.0]  # [vx, vy]
        >>> # Allocating version:
        >>> flux = sam.upwind(u, velocity)
        >>> # No-allocation version:
        >>> flux = sam.ScalarField2D("flux", mesh, 0.0)
        >>> sam.upwind(u, velocity, output=flux)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc");

    // Bind 3D upwind operator - std::array version
    m.def("upwind",
          &upwind_3d,
          py::arg("field"),
          py::arg("velocity"),
          py::arg("output") = py::none(),
          R"pbdoc(
        Upwind operator for 3D advection (std::array version).

        Parameters
        ----------
        field : ScalarField3D
            Input scalar field
        velocity : std::array<double, 3>
            3D velocity vector [vx, vy, vz]
        output : ScalarField3D, optional
            Pre-allocated output field for no-allocation mode

        Returns
        -------
        ScalarField3D
            Field containing upwind flux values
        )pbdoc");

    // Bind 3D upwind operator - Python sequence version (more convenient)
    m.def("upwind",
          &upwind_3d_py,
          py::arg("field"),
          py::arg("velocity"),
          py::arg("output") = py::none(),
          R"pbdoc(
        Upwind operator for 3D advection.

        Computes the upwind flux for a scalar field in 3D.

        Parameters
        ----------
        field : ScalarField3D
            Input scalar field
        velocity : sequence of float
            3D velocity vector [vx, vy, vz] (list or tuple)
        output : ScalarField3D, optional
            Pre-allocated output field for no-allocation mode

        Returns
        -------
        ScalarField3D
            Field containing upwind flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh3D(box, config)
        >>> u = sam.ScalarField3D("u", mesh)
        >>> velocity = [1.0, 1.0, 0.0]  # [vx, vy, vz]
        >>> # Allocating version:
        >>> flux = sam.upwind(u, velocity)
        >>> # No-allocation version:
        >>> flux = sam.ScalarField3D("flux", mesh, 0.0)
        >>> sam.upwind(u, velocity, output=flux)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc");

    // ============================================================
    // WENO5 Convection operators
    // ============================================================

    // ------------------------------------------------------------
    // Non-linear WENO5 (Burgers equation): f(u) = u^2 or u(d)*u
    // ------------------------------------------------------------

    // 1D non-linear WENO5
    m.def("make_convection_weno5",
          &convection_weno5_nonlin_1d,
          py::arg("field"),
          R"pbdoc(
        WENO5 convection operator for 1D Burgers equation (nonlinear).

        5th order Weighted Essentially Non-Oscillatory scheme for nonlinear convection.
        Flux: f(u) = u^2 (scalar) or f(u) = u(d)*u (vector)

        Parameters
        ----------
        field : ScalarField1D
            Input scalar field

        Returns
        -------
        ScalarField1D
            New field containing convection flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh1D(box, config)
        >>> u = sam.ScalarField1D("u", mesh)
        >>> flux = sam.make_convection_weno5(u)
        >>> # Use in time step: unp1 = u - dt * flux
        >>> # For Burgers: flux = u^2/2, so use: unp1 = u - dt * flux
        )pbdoc");

    // 2D non-linear WENO5
    m.def("make_convection_weno5",
          &convection_weno5_nonlin_2d,
          py::arg("field"),
          R"pbdoc(
        WENO5 convection operator for 2D Burgers equation (nonlinear).

        5th order Weighted Essentially Non-Oscillatory scheme for nonlinear convection.
        Flux: f(u) = u^2 (scalar) or f(u) = u(d)*u (vector)

        Parameters
        ----------
        field : ScalarField2D
            Input scalar field

        Returns
        -------
        ScalarField2D
            New field containing convection flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh2D(box, config)
        >>> u = sam.ScalarField2D("u", mesh)
        >>> flux = sam.make_convection_weno5(u)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc");

    // 3D non-linear WENO5
    m.def("make_convection_weno5",
          &convection_weno5_nonlin_3d,
          py::arg("field"),
          R"pbdoc(
        WENO5 convection operator for 3D Burgers equation (nonlinear).

        5th order Weighted Essentially Non-Oscillatory scheme for nonlinear convection.
        Flux: f(u) = u^2 (scalar) or f(u) = u(d)*u (vector)

        Parameters
        ----------
        field : ScalarField3D
            Input scalar field

        Returns
        -------
        ScalarField3D
            New field containing convection flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh3D(box, config)
        >>> u = sam.ScalarField3D("u", mesh)
        >>> flux = sam.make_convection_weno5(u)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc");

    // ------------------------------------------------------------
    // Linear WENO5 with constant velocity: f(u) = velocity * u
    // ------------------------------------------------------------

    // 1D linear WENO5
    m.def("make_convection_weno5",
          &convection_weno5_linear_1d,
          py::arg("field"),
          py::arg("velocity"),
          R"pbdoc(
        WENO5 convection operator for 1D linear advection with constant velocity.

        5th order Weighted Essentially Non-Oscillatory scheme for linear convection.
        Flux: f(u) = velocity * u

        Parameters
        ----------
        field : ScalarField1D
            Input scalar field
        velocity : float
            Advection velocity (scalar for 1D)

        Returns
        -------
        ScalarField1D
            New field containing convection flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh1D(box, config)
        >>> u = sam.ScalarField1D("u", mesh)
        >>> flux = sam.make_convection_weno5(u, 1.0)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc");

    // 2D linear WENO5 - std::array version
    m.def("make_convection_weno5",
          &convection_weno5_linear_2d,
          py::arg("field"),
          py::arg("velocity"),
          R"pbdoc(
        WENO5 convection operator for 2D linear advection with constant velocity (std::array version).

        Parameters
        ----------
        field : ScalarField2D
            Input scalar field
        velocity : std::array<double, 2>
            2D velocity vector [vx, vy]

        Returns
        -------
        ScalarField2D
            New field containing convection flux values
        )pbdoc");

    // 2D linear WENO5 - Python sequence version
    m.def("make_convection_weno5",
          &convection_weno5_linear_2d_py,
          py::arg("field"),
          py::arg("velocity"),
          R"pbdoc(
        WENO5 convection operator for 2D linear advection with constant velocity.

        5th order Weighted Essentially Non-Oscillatory scheme for linear convection.
        Flux: f(u) = velocity · u

        Parameters
        ----------
        field : ScalarField2D
            Input scalar field
        velocity : sequence of float
            2D velocity vector [vx, vy] (list or tuple)

        Returns
        -------
        ScalarField2D
            New field containing convection flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh2D(box, config)
        >>> u = sam.ScalarField2D("u", mesh)
        >>> velocity = [1.0, 1.0]  # [vx, vy]
        >>> flux = sam.make_convection_weno5(u, velocity)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc");

    // 3D linear WENO5 - std::array version
    m.def("make_convection_weno5",
          &convection_weno5_linear_3d,
          py::arg("field"),
          py::arg("velocity"),
          R"pbdoc(
        WENO5 convection operator for 3D linear advection with constant velocity (std::array version).

        Parameters
        ----------
        field : ScalarField3D
            Input scalar field
        velocity : std::array<double, 3>
            3D velocity vector [vx, vy, vz]

        Returns
        -------
        ScalarField3D
            New field containing convection flux values
        )pbdoc");

    // 3D linear WENO5 - Python sequence version
    m.def("make_convection_weno5",
          &convection_weno5_linear_3d_py,
          py::arg("field"),
          py::arg("velocity"),
          R"pbdoc(
        WENO5 convection operator for 3D linear advection with constant velocity.

        5th order Weighted Essentially Non-Oscillatory scheme for linear convection.
        Flux: f(u) = velocity · u

        Parameters
        ----------
        field : ScalarField3D
            Input scalar field
        velocity : sequence of float
            3D velocity vector [vx, vy, vz] (list or tuple)

        Returns
        -------
        ScalarField3D
            New field containing convection flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh3D(box, config)
        >>> u = sam.ScalarField3D("u", mesh)
        >>> velocity = [1.0, 1.0, 0.0]  # [vx, vy, vz]
        >>> flux = sam.make_convection_weno5(u, velocity)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc");

    // ============================================================
    // WENO5 Convection operators for VectorField (Burgers equation)
    // ============================================================

    // 2D non-linear WENO5 for VectorField2D_2 (Burgers 2D)
    m.def("make_convection_weno5",
          &convection_weno5_nonlin_vector_2d,
          py::arg("field"),
          R"pbdoc(
        WENO5 convection operator for 2D Burgers equation (nonlinear, vector field).

        5th order Weighted Essentially Non-Oscillatory scheme for nonlinear convection.
        Solves the vector Burgers equation: ∂u/∂t + u·∇u = 0
        where u = [u, v] is the velocity vector field.

        Flux: F(u) = u ⊗ u = [[u^2, uv], [uv, v^2]]

        Parameters
        ----------
        field : VectorField2D_2
            Input vector field [u, v] with 2 components

        Returns
        -------
        VectorField2D_2
            New field containing convection flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh2D(box, config)
        >>> u = sam.VectorField2D_2("u", mesh, 0.0)
        >>> # Initialize u with velocity field
        >>> flux = sam.make_convection_weno5(u)
        >>> # RK3 time stepping
        >>> u1 = u - dt * flux
        >>> u2 = 3./4 * u + 1./4 * (u1 - dt * sam.make_convection_weno5(u1))
        >>> unp1 = 1./3 * u + 2./3 * (u2 - dt * sam.make_convection_weno5(u2))
        )pbdoc");

    // 3D non-linear WENO5 for VectorField3D_3 (Burgers 3D)
    m.def("make_convection_weno5",
          &convection_weno5_nonlin_vector_3d,
          py::arg("field"),
          R"pbdoc(
        WENO5 convection operator for 3D Burgers equation (nonlinear, vector field).

        5th order Weighted Essentially Non-Oscillatory scheme for nonlinear convection.
        Solves the vector Burgers equation: ∂u/∂t + u·∇u = 0
        where u = [u, v, w] is the velocity vector field.

        Flux: F(u) = u ⊗ u (tensor product)

        Parameters
        ----------
        field : VectorField3D_3
            Input vector field [u, v, w] with 3 components

        Returns
        -------
        VectorField3D_3
            New field containing convection flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh3D(box, config)
        >>> u = sam.VectorField3D_3("u", mesh, 0.0)
        >>> # Initialize u with velocity field
        >>> flux = sam.make_convection_weno5(u)
        >>> # RK3 time stepping
        >>> u1 = u - dt * flux
        >>> u2 = 3./4 * u + 1./4 * (u1 - dt * sam.make_convection_weno5(u1))
        >>> unp1 = 1./3 * u + 2./3 * (u2 - dt * sam.make_convection_weno5(u2))
        )pbdoc");

    // ============================================================
    // Linear WENO5 with VectorField velocity: f(u) = velocity(x) * u
    // ============================================================

    // 2D ScalarField with VectorField2D_2 velocity
    m.def("make_convection_weno5",
          &convection_weno5_vectorfield_2d,
          py::arg("field"),
          py::arg("velocity"),
          R"pbdoc(
        WENO5 convection operator for 2D linear advection with VectorField velocity.

        5th order Weighted Essentially Non-Oscillatory scheme for linear convection
        with spatially varying velocity field.
        Flux: f(u) = velocity(x) · u

        Parameters
        ----------
        field : ScalarField2D
            Input scalar field
        velocity : VectorField2D_2
            Velocity field [u, v] (can vary in space)

        Returns
        -------
        ScalarField2D
            New field containing convection flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh2D(domain, config)
        >>> velocity = sam.VectorField2D_2("vel", mesh, 0.0)
        >>> # Initialize velocity field (constant or space-dependent)
        >>> velocity = sam.make_vector_field(mesh, "velocity",
        ...     lambda center: [1.0, -1.0], 2)
        >>> u = sam.ScalarField2D("u", mesh, 0.0)
        >>> flux = sam.make_convection_weno5(u, velocity)
        >>> # Use in time step: unp1 = u - dt * flux

        Notes
        -----
        This overload is useful for:
        - Obstacle problems where velocity varies near boundaries
        - Complex flow fields with spatial variation
        - Consistent velocity treatment across mesh adaptation
        )pbdoc");

    // 3D ScalarField with VectorField3D_3 velocity
    m.def("make_convection_weno5",
          &convection_weno5_vectorfield_3d,
          py::arg("field"),
          py::arg("velocity"),
          R"pbdoc(
        WENO5 convection operator for 3D linear advection with VectorField velocity.

        Similar to 2D version but for 3D meshes with VectorField3D_3 velocity.

        Flux: f(u) = velocity(x) · u

        Parameters
        ----------
        field : ScalarField3D
            Input scalar field
        velocity : VectorField3D_3
            Velocity field [u, v, w] (can vary in space)

        Returns
        -------
        ScalarField3D
            New field containing convection flux values
        )pbdoc");

    // ============================================================
    // Diffusion operators (order 2)
    // ============================================================

    // 1D diffusion
    m.def("make_diffusion_order2",
          &diffusion_order2_1d,
          py::arg("field"),
          py::arg("coefficient") = 1.0,
          R"pbdoc(
        Diffusion operator for 1D scalar fields.

        Computes -coefficient * Laplacian(field), where Laplacian is the
        positive (mathematical) Laplacian operator.

        Parameters
        ----------
        field : ScalarField1D
            Input scalar field
        coefficient : float, optional
            Diffusion coefficient (default: 1.0)

        Returns
        -------
        ScalarField1D
            New field containing -coefficient * Laplacian(field)

        Examples
        --------
        >>> import sampai as sam
        >>> mesh = sam.mesh.make(box, min_level=3, max_level=3)
        >>> u = sam.field.scalar("u", mesh)
        >>> diff_u = sam.make_diffusion_order2(u, 0.5)  # -0.5 * Laplacian(u)
        >>> # Explicit time stepping: unp1 = u + dt * diff_u
        )pbdoc");

    // 2D diffusion
    m.def("make_diffusion_order2",
          &diffusion_order2_2d,
          py::arg("field"),
          py::arg("coefficient") = 1.0,
          R"pbdoc(
        Diffusion operator for 2D scalar fields.

        Computes -coefficient * Laplacian(field).

        Parameters
        ----------
        field : ScalarField2D
            Input scalar field
        coefficient : float, optional
            Diffusion coefficient (default: 1.0)

        Returns
        -------
        ScalarField2D
            New field containing -coefficient * Laplacian(field)

        Examples
        --------
        >>> mesh = sam.mesh.make(box, min_level=3, max_level=3)
        >>> u = sam.field.scalar("u", mesh)
        >>> diff_u = sam.make_diffusion_order2(u, 0.5)
        )pbdoc");

    // 3D diffusion
    m.def("make_diffusion_order2",
          &diffusion_order2_3d,
          py::arg("field"),
          py::arg("coefficient") = 1.0,
          R"pbdoc(
        Diffusion operator for 3D scalar fields.

        Computes -coefficient * Laplacian(field).

        Parameters
        ----------
        field : ScalarField3D
            Input scalar field
        coefficient : float, optional
            Diffusion coefficient (default: 1.0)

        Returns
        -------
        ScalarField3D
            New field containing -coefficient * Laplacian(field)
        )pbdoc");

    // ============================================================
    // Laplacian operators (order 2)
    // ============================================================

    // 1D Laplacian
    m.def("make_laplacian_order2",
          &laplacian_order2_1d,
          py::arg("field"),
          R"pbdoc(
        Laplacian operator for 1D scalar fields.

        Computes the positive (mathematical) Laplacian: Delta(field).
        Equivalent to -make_diffusion_order2(field).

        Parameters
        ----------
        field : ScalarField1D
            Input scalar field

        Returns
        -------
        ScalarField1D
            New field containing Laplacian(field)

        Examples
        --------
        >>> import sampai as sam
        >>> mesh = sam.mesh.make(box, min_level=3, max_level=3)
        >>> u = sam.field.scalar("u", mesh)
        >>> lap_u = sam.make_laplacian_order2(u)  # Delta(u)
        )pbdoc");

    // 2D Laplacian
    m.def("make_laplacian_order2",
          &laplacian_order2_2d,
          py::arg("field"),
          R"pbdoc(
        Laplacian operator for 2D scalar fields.

        Computes Delta(field).

        Parameters
        ----------
        field : ScalarField2D
            Input scalar field

        Returns
        -------
        ScalarField2D
            New field containing Laplacian(field)
        )pbdoc");

    // 3D Laplacian
    m.def("make_laplacian_order2",
          &laplacian_order2_3d,
          py::arg("field"),
          R"pbdoc(
        Laplacian operator for 3D scalar fields.

        Computes Delta(field).

        Parameters
        ----------
        field : ScalarField3D
            Input scalar field

        Returns
        -------
        ScalarField3D
            New field containing Laplacian(field)
        )pbdoc");

    // ============================================================
    // Gradient operators (order 2)
    // ============================================================
    // NOTE: 1D gradient is not supported due to VectorField1D_1 incompatibility with explicit scheme

    // 2D Gradient
    m.def("make_gradient_order2",
          &gradient_order2_2d,
          py::arg("field"),
          R"pbdoc(
        Gradient operator for 2D scalar fields.

        Computes the gradient of a scalar field.
        Returns a VectorField2D_2 (2 components in 2D space).

        Parameters
        ----------
        field : ScalarField2D
            Input scalar field

        Returns
        -------
        VectorField2D_2
            Vector field containing grad(field) = [du/dx, du/dy]
        )pbdoc");

    // 3D Gradient
    m.def("make_gradient_order2",
          &gradient_order2_3d,
          py::arg("field"),
          R"pbdoc(
        Gradient operator for 3D scalar fields.

        Computes the gradient of a scalar field.
        Returns a VectorField3D_3 (3 components in 3D space).

        Parameters
        ----------
        field : ScalarField3D
            Input scalar field

        Returns
        -------
        VectorField3D_3
            Vector field containing grad(field) = [du/dx, du/dy, du/dz]
        )pbdoc");

    // ============================================================
    // Divergence operators (order 2)
    // ============================================================
    // NOTE: 1D divergence is not supported due to VectorField1D_1 incompatibility with explicit scheme

    // 2D Divergence
    m.def("make_divergence_order2",
          &divergence_order2_2d,
          py::arg("field"),
          R"pbdoc(
        Divergence operator for 2D vector fields.

        Computes the divergence of a vector field.
        Input: VectorField2D_2, Output: ScalarField2D.

        Parameters
        ----------
        field : VectorField2D_2
            Input vector field (2 components in 2D space)

        Returns
        -------
        ScalarField2D
            Scalar field containing div(field) = dv_x/dx + dv_y/dy
        )pbdoc");

    // 3D Divergence
    m.def("make_divergence_order2",
          &divergence_order2_3d,
          py::arg("field"),
          R"pbdoc(
        Divergence operator for 3D vector fields.

        Computes the divergence of a vector field.
        Input: VectorField3D_3, Output: ScalarField3D.

        Parameters
        ----------
        field : VectorField3D_3
            Input vector field (3 components in 3D space)

        Returns
        -------
        ScalarField3D
            Scalar field containing div(field) = dv_x/dx + dv_y/dy + dv_z/dz
        )pbdoc");

    // ============================================================
    // Create operators submodule for better organization
    // ============================================================

    // Create the operators submodule
    py::module_ operators = m.def_submodule("operators", "Finite volume operators for AMR");

    // Reference all operator functions in the submodule
    // This maintains backward compatibility (operators still in main module)
    // while also providing them in the organized submodule

    // Upwind operators (with optional output parameter for no-allocation)
    operators.attr("upwind") = m.attr("upwind");

    // WENO5 convection operators
    operators.attr("make_convection_weno5") = m.attr("make_convection_weno5");

    // Alias for shorter name (without 'make_' prefix, more Pythonic)
    operators.attr("convection_weno5") = m.attr("make_convection_weno5");

    // Diffusion operators
    operators.attr("make_diffusion_order2") = m.attr("make_diffusion_order2");
    operators.attr("diffusion_order2") = m.attr("make_diffusion_order2");  // Alias

    // Laplacian operators
    operators.attr("make_laplacian_order2") = m.attr("make_laplacian_order2");
    operators.attr("laplacian_order2") = m.attr("make_laplacian_order2");  // Alias

    // Gradient operators
    operators.attr("make_gradient_order2") = m.attr("make_gradient_order2");
    operators.attr("gradient_order2") = m.attr("make_gradient_order2");  // Alias

    // Divergence operators
    operators.attr("make_divergence_order2") = m.attr("make_divergence_order2");
    operators.attr("divergence_order2") = m.attr("make_divergence_order2");  // Alias
}
