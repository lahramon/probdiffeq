"""Recipes for ODE filters.

Learning about the inner workings of an ODE filter is a little too much?
We hear ya -- tt can indeed get quite complicated.
Therefore, here we provide some recipes that create our favourite,
time-tested ODE filter versions.
We still recommend to build an ODE filter yourself,
but until you do so, use one of ours.

!!! danger "Highly experimental"

    Don't trust that this module survives.
    It adds nothing to the actual code, it just makes it easier to use.
    Tomorrow, this module might go again.

"""
from odefilter import controls, information, odefilters, strategies
from odefilter.implementations import dense, isotropic

ATOL_DEFAULTS = 1e-6
RTOL_DEFAULTS = 1e-3


def dynamic_isotropic_ekf0(num_derivatives, atol=ATOL_DEFAULTS, rtol=RTOL_DEFAULTS):
    """Construct the equivalent of an explicit solver with an isotropic covariance \
    structure, dynamic calibration, and optimised for terminal-value simulation.

    Suitable for high-dimensional, non-stiff problems.
    """
    information_op = information.IsotropicEK0FirstOrder()
    implementation = isotropic.IsotropicImplementation.from_num_derivatives(
        num_derivatives=num_derivatives
    )
    strategy = strategies.DynamicFilter(
        implementation=implementation, information=information_op
    )
    control = controls.ProportionalIntegral()
    return odefilters.AdaptiveODEFilter(
        strategy=strategy,
        control=control,
        atol=atol,
        rtol=rtol,
        error_order=num_derivatives + 1,
    )


def dynamic_isotropic_eks0(num_derivatives, atol=ATOL_DEFAULTS, rtol=RTOL_DEFAULTS):
    """Construct the equivalent of an explicit solver with an isotropic covariance \
    structure and dynamic calibration.

    Suitable for high-dimensional, non-stiff problems.
    """
    information_op = information.IsotropicEK0FirstOrder()
    implementation = isotropic.IsotropicImplementation.from_num_derivatives(
        num_derivatives=num_derivatives
    )
    strategy = strategies.DynamicSmoother(
        implementation=implementation, information=information_op
    )
    control = controls.ProportionalIntegral()
    return odefilters.AdaptiveODEFilter(
        strategy=strategy,
        control=control,
        atol=atol,
        rtol=rtol,
        error_order=num_derivatives + 1,
    )


def dynamic_ekf1(
    num_derivatives, ode_dimension, atol=ATOL_DEFAULTS, rtol=RTOL_DEFAULTS
):
    """Construct the equivalent of a semi-implicit solver with dynamic calibration.

    Suitable for low-dimensional, stiff problems.
    """
    information_op = information.EK1FirstOrder(ode_dimension=ode_dimension)
    implementation = dense.DenseImplementation.from_num_derivatives(
        num_derivatives=num_derivatives, ode_dimension=ode_dimension
    )
    strategy = strategies.DynamicFilter(
        implementation=implementation, information=information_op
    )
    control = controls.ProportionalIntegral()
    return odefilters.AdaptiveODEFilter(
        strategy=strategy,
        control=control,
        atol=atol,
        rtol=rtol,
        error_order=num_derivatives + 1,
    )