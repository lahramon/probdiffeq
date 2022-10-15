"""Test cases: Solver recipes.

All ODE test problems will be two-dimensional.
"""
from pytest_cases import case

from odefilter import recipes


@case(tags=["terminal_value", "checkpoint"])
def solver_dynamic_isotropic_fixpt_eks0():
    return recipes.dynamic_isotropic_fixpt_eks0(num_derivatives=3, atol=1e-4, rtol=1e-4)


@case(tags=("terminal_value", "solve"))
def solver_dynamic_isotropic_eks0():
    return recipes.dynamic_isotropic_eks0(num_derivatives=3, atol=1e-4, rtol=1e-4)


@case(tags=("terminal_value", "solve", "checkpoint"))
def solver_dynamic_isotropic_ekf0():
    return recipes.dynamic_isotropic_ekf0(num_derivatives=3, atol=1e-4, rtol=1e-4)


@case(tags=("terminal_value", "solve", "checkpoint"))
def solver_dynamic_ekf1():
    return recipes.dynamic_ekf1(
        num_derivatives=3, ode_dimension=2, atol=1e-4, rtol=1e-4
    )