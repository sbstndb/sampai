// Copyright 2025 the sampai's authors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <samurai/field.hpp>
#include <samurai/algorithm.hpp>
#include "common_types.hpp"

namespace samurai::python::arithmetic {

// Import type aliases from common_types.hpp for convenience
using namespace samurai::python::bindings;

// ============================================================================
// Field arithmetic operations (NumPy-style, immediate evaluation)
// ============================================================================
//
// This module provides simple arithmetic operations for ScalarField and
// VectorField that do NOT rely on the expression template system.
// All operations are evaluated immediately and return new fields.
//
// Design principles:
// - Simple, explicit functions
// - Immediate evaluation (no lazy expressions)
// - Works correctly after mesh adaptation
// - Ghost cells are set to 0.0 by default (not garbage)
// ============================================================================

// -------------------------------------------------------------------------
// ScalarField arithmetic - field +/-/* scalar operations
// -------------------------------------------------------------------------

template <std::size_t dim, class value_t = double>
ScalarField<dim> add(const ScalarField<dim>& field, value_t scalar)
{
    auto& mesh = const_cast<typename ScalarField<dim>::mesh_t&>(field.mesh());
    auto result = samurai::make_scalar_field<value_t>(field.name() + "_add", mesh, 0.0);

    samurai::for_each_interval(mesh,
        [&field, &result, scalar](std::size_t level, const auto& interval, const auto& index)
        {
            result(level, interval, index) = field(level, interval, index) + scalar;
        });

    return result;
}

template <std::size_t dim, class value_t = double>
ScalarField<dim> sub(const ScalarField<dim>& field, value_t scalar)
{
    auto& mesh = const_cast<typename ScalarField<dim>::mesh_t&>(field.mesh());
    auto result = samurai::make_scalar_field<value_t>(field.name() + "_sub", mesh, 0.0);

    samurai::for_each_interval(mesh,
        [&field, &result, scalar](std::size_t level, const auto& interval, const auto& index)
        {
            result(level, interval, index) = field(level, interval, index) - scalar;
        });

    return result;
}

template <std::size_t dim, class value_t = double>
ScalarField<dim> mul(const ScalarField<dim>& field, value_t scalar)
{
    auto& mesh = const_cast<typename ScalarField<dim>::mesh_t&>(field.mesh());
    auto result = samurai::make_scalar_field<value_t>(field.name() + "_mul", mesh, 0.0);

    samurai::for_each_interval(mesh,
        [&field, &result, scalar](std::size_t level, const auto& interval, const auto& index)
        {
            result(level, interval, index) = field(level, interval, index) * scalar;
        });

    return result;
}

template <std::size_t dim, class value_t = double>
ScalarField<dim> div(const ScalarField<dim>& field, value_t scalar)
{
    auto& mesh = const_cast<typename ScalarField<dim>::mesh_t&>(field.mesh());
    auto result = samurai::make_scalar_field<value_t>(field.name() + "_div", mesh, 0.0);

    samurai::for_each_interval(mesh,
        [&field, &result, scalar](std::size_t level, const auto& interval, const auto& index)
        {
            result(level, interval, index) = field(level, interval, index) / scalar;
        });

    return result;
}

// -------------------------------------------------------------------------
// ScalarField arithmetic - scalar - field (reverse operations)
// -------------------------------------------------------------------------

template <std::size_t dim, class value_t = double>
ScalarField<dim> rsub(value_t scalar, const ScalarField<dim>& field)
{
    auto& mesh = const_cast<typename ScalarField<dim>::mesh_t&>(field.mesh());
    auto result = samurai::make_scalar_field<value_t>("scalar_sub", mesh, 0.0);

    samurai::for_each_interval(mesh,
        [&field, &result, scalar](std::size_t level, const auto& interval, const auto& index)
        {
            result(level, interval, index) = scalar - field(level, interval, index);
        });

    return result;
}

// -------------------------------------------------------------------------
// ScalarField arithmetic - field +/-/* field operations
// -------------------------------------------------------------------------

template <std::size_t dim, class value_t = double>
ScalarField<dim> add(const ScalarField<dim>& field1, const ScalarField<dim>& field2)
{
    auto& mesh = const_cast<typename ScalarField<dim>::mesh_t&>(field1.mesh());
    auto result = samurai::make_scalar_field<value_t>(field1.name() + "_add", mesh, 0.0);

    samurai::for_each_interval(mesh,
        [&field1, &field2, &result](std::size_t level, const auto& interval, const auto& index)
        {
            result(level, interval, index) = field1(level, interval, index) + field2(level, interval, index);
        });

    return result;
}

template <std::size_t dim, class value_t = double>
ScalarField<dim> sub(const ScalarField<dim>& field1, const ScalarField<dim>& field2)
{
    auto& mesh = const_cast<typename ScalarField<dim>::mesh_t&>(field1.mesh());
    auto result = samurai::make_scalar_field<value_t>(field1.name() + "_sub", mesh, 0.0);

    samurai::for_each_interval(mesh,
        [&field1, &field2, &result](std::size_t level, const auto& interval, const auto& index)
        {
            result(level, interval, index) = field1(level, interval, index) - field2(level, interval, index);
        });

    return result;
}

template <std::size_t dim, class value_t = double>
ScalarField<dim> mul(const ScalarField<dim>& field1, const ScalarField<dim>& field2)
{
    auto& mesh = const_cast<typename ScalarField<dim>::mesh_t&>(field1.mesh());
    auto result = samurai::make_scalar_field<value_t>(field1.name() + "_mul", mesh, 0.0);

    samurai::for_each_interval(mesh,
        [&field1, &field2, &result](std::size_t level, const auto& interval, const auto& index)
        {
            result(level, interval, index) = field1(level, interval, index) * field2(level, interval, index);
        });

    return result;
}

template <std::size_t dim, class value_t = double>
ScalarField<dim> div(const ScalarField<dim>& field1, const ScalarField<dim>& field2)
{
    auto& mesh = const_cast<typename ScalarField<dim>::mesh_t&>(field1.mesh());
    auto result = samurai::make_scalar_field<value_t>(field1.name() + "_div", mesh, 0.0);

    samurai::for_each_interval(mesh,
        [&field1, &field2, &result](std::size_t level, const auto& interval, const auto& index)
        {
            result(level, interval, index) = field1(level, interval, index) / field2(level, interval, index);
        });

    return result;
}

// -------------------------------------------------------------------------
// ScalarField in-place operations (PyTorch-style with underscore)
// -------------------------------------------------------------------------

template <std::size_t dim, class value_t = double>
void add_in_place(ScalarField<dim>& field, value_t scalar)
{
    samurai::for_each_interval(field.mesh(),
        [&field, scalar](std::size_t level, const auto& interval, const auto& index)
        {
            field(level, interval, index) += scalar;
        });
    field.ghosts_updated() = false;
}

template <std::size_t dim, class value_t = double>
void sub_in_place(ScalarField<dim>& field, value_t scalar)
{
    samurai::for_each_interval(field.mesh(),
        [&field, scalar](std::size_t level, const auto& interval, const auto& index)
        {
            field(level, interval, index) -= scalar;
        });
    field.ghosts_updated() = false;
}

template <std::size_t dim, class value_t = double>
void mul_in_place(ScalarField<dim>& field, value_t scalar)
{
    samurai::for_each_interval(field.mesh(),
        [&field, scalar](std::size_t level, const auto& interval, const auto& index)
        {
            field(level, interval, index) *= scalar;
        });
    field.ghosts_updated() = false;
}

template <std::size_t dim, class value_t = double>
void div_in_place(ScalarField<dim>& field, value_t scalar)
{
    samurai::for_each_interval(field.mesh(),
        [&field, scalar](std::size_t level, const auto& interval, const auto& index)
        {
            field(level, interval, index) /= scalar;
        });
    field.ghosts_updated() = false;
}

// -------------------------------------------------------------------------
// ScalarField in-place operations with field operand
// -------------------------------------------------------------------------

template <std::size_t dim, class value_t = double>
void add_in_place(ScalarField<dim>& field1, const ScalarField<dim>& field2)
{
    samurai::for_each_interval(field1.mesh(),
        [&field1, &field2](std::size_t level, const auto& interval, const auto& index)
        {
            field1(level, interval, index) += field2(level, interval, index);
        });
    field1.ghosts_updated() = false;
}

template <std::size_t dim, class value_t = double>
void sub_in_place(ScalarField<dim>& field1, const ScalarField<dim>& field2)
{
    samurai::for_each_interval(field1.mesh(),
        [&field1, &field2](std::size_t level, const auto& interval, const auto& index)
        {
            field1(level, interval, index) -= field2(level, interval, index);
        });
    field1.ghosts_updated() = false;
}

// -------------------------------------------------------------------------
// Clone operation (deep copy)
// -------------------------------------------------------------------------

template <std::size_t dim, class value_t = double>
ScalarField<dim> clone(const ScalarField<dim>& field)
{
    auto& mesh = const_cast<typename ScalarField<dim>::mesh_t&>(field.mesh());
    auto result = samurai::make_scalar_field<value_t>(field.name() + "_clone", mesh);

    samurai::for_each_interval(mesh,
        [&field, &result](std::size_t level, const auto& interval, const auto& index)
        {
            result(level, interval, index) = field(level, interval, index);
        });

    return result;
}

// -------------------------------------------------------------------------
// Copy operation (copy from source to destination)
// -------------------------------------------------------------------------

template <std::size_t dim, class value_t = double>
void copy_to(const ScalarField<dim>& src, ScalarField<dim>& dest)
{
    samurai::for_each_interval(src.mesh(),
        [&src, &dest](std::size_t level, const auto& interval, const auto& index)
        {
            dest(level, interval, index) = src(level, interval, index);
        });
    dest.ghosts_updated() = src.ghosts_updated();
}

// -------------------------------------------------------------------------
// VectorField arithmetic operations
// -------------------------------------------------------------------------

template <std::size_t dim, std::size_t n_comp, bool SOA, class value_t = double>
VectorField<dim, n_comp, SOA> add(const VectorField<dim, n_comp, SOA>& field, value_t scalar)
{
    auto& mesh = const_cast<typename VectorField<dim, n_comp, SOA>::mesh_t&>(field.mesh());
    auto result = samurai::make_vector_field<value_t, n_comp, SOA>(field.name() + "_add", mesh, 0.0);

    samurai::for_each_interval(mesh,
        [&field, &result, scalar](std::size_t level, const auto& interval, const auto& index)
        {
            result(level, interval, index) = field(level, interval, index) + scalar;
        });

    return result;
}

template <std::size_t dim, std::size_t n_comp, bool SOA, class value_t = double>
VectorField<dim, n_comp, SOA> sub(const VectorField<dim, n_comp, SOA>& field, value_t scalar)
{
    auto& mesh = const_cast<typename VectorField<dim, n_comp, SOA>::mesh_t&>(field.mesh());
    auto result = samurai::make_vector_field<value_t, n_comp, SOA>(field.name() + "_sub", mesh, 0.0);

    samurai::for_each_interval(mesh,
        [&field, &result, scalar](std::size_t level, const auto& interval, const auto& index)
        {
            result(level, interval, index) = field(level, interval, index) - scalar;
        });

    return result;
}

template <std::size_t dim, std::size_t n_comp, bool SOA, class value_t = double>
VectorField<dim, n_comp, SOA> mul(const VectorField<dim, n_comp, SOA>& field, value_t scalar)
{
    auto& mesh = const_cast<typename VectorField<dim, n_comp, SOA>::mesh_t&>(field.mesh());
    auto result = samurai::make_vector_field<value_t, n_comp, SOA>(field.name() + "_mul", mesh, 0.0);

    samurai::for_each_interval(mesh,
        [&field, &result, scalar](std::size_t level, const auto& interval, const auto& index)
        {
            result(level, interval, index) = field(level, interval, index) * scalar;
        });

    return result;
}

template <std::size_t dim, std::size_t n_comp, bool SOA, class value_t = double>
VectorField<dim, n_comp, SOA> div(const VectorField<dim, n_comp, SOA>& field, value_t scalar)
{
    auto& mesh = const_cast<typename VectorField<dim, n_comp, SOA>::mesh_t&>(field.mesh());
    auto result = samurai::make_vector_field<value_t, n_comp, SOA>(field.name() + "_div", mesh, 0.0);

    samurai::for_each_interval(mesh,
        [&field, &result, scalar](std::size_t level, const auto& interval, const auto& index)
        {
            result(level, interval, index) = field(level, interval, index) / scalar;
        });

    return result;
}

// -------------------------------------------------------------------------
// VectorField arithmetic - scalar - field (reverse operations)
// -------------------------------------------------------------------------

template <std::size_t dim, std::size_t n_comp, bool SOA, class value_t = double>
VectorField<dim, n_comp, SOA> rsub(value_t scalar, const VectorField<dim, n_comp, SOA>& field)
{
    auto& mesh = const_cast<typename VectorField<dim, n_comp, SOA>::mesh_t&>(field.mesh());
    auto result = samurai::make_vector_field<value_t, n_comp, SOA>("scalar_sub", mesh, 0.0);

    samurai::for_each_interval(mesh,
        [&field, &result, scalar](std::size_t level, const auto& interval, const auto& index)
        {
            result(level, interval, index) = scalar - field(level, interval, index);
        });

    return result;
}

// VectorField in-place operations
template <std::size_t dim, std::size_t n_comp, bool SOA, class value_t = double>
void add_in_place(VectorField<dim, n_comp, SOA>& field, value_t scalar)
{
    samurai::for_each_interval(field.mesh(),
        [&field, scalar](std::size_t level, const auto& interval, const auto& index)
        {
            field(level, interval, index) += scalar;
        });
    field.ghosts_updated() = false;
}

template <std::size_t dim, std::size_t n_comp, bool SOA, class value_t = double>
void sub_in_place(VectorField<dim, n_comp, SOA>& field, value_t scalar)
{
    samurai::for_each_interval(field.mesh(),
        [&field, scalar](std::size_t level, const auto& interval, const auto& index)
        {
            field(level, interval, index) -= scalar;
        });
    field.ghosts_updated() = false;
}

template <std::size_t dim, std::size_t n_comp, bool SOA, class value_t = double>
void mul_in_place(VectorField<dim, n_comp, SOA>& field, value_t scalar)
{
    samurai::for_each_interval(field.mesh(),
        [&field, scalar](std::size_t level, const auto& interval, const auto& index)
        {
            field(level, interval, index) *= scalar;
        });
    field.ghosts_updated() = false;
}

template <std::size_t dim, std::size_t n_comp, bool SOA, class value_t = double>
void div_in_place(VectorField<dim, n_comp, SOA>& field, value_t scalar)
{
    samurai::for_each_interval(field.mesh(),
        [&field, scalar](std::size_t level, const auto& interval, const auto& index)
        {
            field(level, interval, index) /= scalar;
        });
    field.ghosts_updated() = false;
}

// VectorField - field operations
template <std::size_t dim, std::size_t n_comp, bool SOA, class value_t = double>
VectorField<dim, n_comp, SOA> add(const VectorField<dim, n_comp, SOA>& field1, const VectorField<dim, n_comp, SOA>& field2)
{
    auto& mesh = const_cast<typename VectorField<dim, n_comp, SOA>::mesh_t&>(field1.mesh());
    auto result = samurai::make_vector_field<value_t, n_comp, SOA>(field1.name() + "_add", mesh, 0.0);

    samurai::for_each_interval(mesh,
        [&field1, &field2, &result](std::size_t level, const auto& interval, const auto& index)
        {
            result(level, interval, index) = field1(level, interval, index) + field2(level, interval, index);
        });

    return result;
}

template <std::size_t dim, std::size_t n_comp, bool SOA, class value_t = double>
VectorField<dim, n_comp, SOA> sub(const VectorField<dim, n_comp, SOA>& field1, const VectorField<dim, n_comp, SOA>& field2)
{
    auto& mesh = const_cast<typename VectorField<dim, n_comp, SOA>::mesh_t&>(field1.mesh());
    auto result = samurai::make_vector_field<value_t, n_comp, SOA>(field1.name() + "_sub", mesh, 0.0);

    samurai::for_each_interval(mesh,
        [&field1, &field2, &result](std::size_t level, const auto& interval, const auto& index)
        {
            result(level, interval, index) = field1(level, interval, index) - field2(level, interval, index);
        });

    return result;
}

} // namespace samurai::python::arithmetic
